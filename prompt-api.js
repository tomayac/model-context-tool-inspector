/**
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// Prompting the browser's built-in on-device model through the Prompt API,
// with the WebMCP tools of the page being inspected.
//
// Tool declarations are built from the tools the page has registered right
// now, so the on-device model is offered the same tools as the ones the
// sidebar lists, however they change as the user navigates.
//
// The declarations carry no execute() callback on purpose: tool calls come
// back as chunks of the response stream and are answered with tool responses,
// which keeps the tool loop on this side, where each call can be routed to the
// frame that registered the tool.

// A small model can keep asking for tools forever, so cap the rounds per turn.
const MAX_TOOL_CALLS = 8;

// What the sidebar hands over: the tools of the page and how to run one, the
// system instruction, where to write, and the trace to record into.
let sidebar;

export function initPromptApi(dependencies) {
  sidebar = dependencies;
}

const isPromptApiSupported = () => 'LanguageModel' in self;

// Tool calls and their responses are interfaces of their own, gated behind the
// same flag as tool use itself, so their absence pinpoints the flag rather
// than the Prompt API.
const isToolUseSupported = () =>
  'LanguageModelToolCall' in self &&
  'LanguageModelToolSuccess' in self &&
  'LanguageModelToolError' in self;

let session, sessionDeclarations;
// The page tool each declared tool name maps back to, since the name the model
// calls is the deduped one.
let toolsByDeclaredName = new Map();
// The text turns of the running conversation. Tools can only be declared when
// a session is created, so a changed tool set means a new session, and this is
// what is replayed into it as `initialPrompts`.
let history = [];

export function endConversation() {
  session?.destroy();
  session = undefined;
  sessionDeclarations = undefined;
  history = [];
}

export async function promptOnDeviceAI(tabId, message) {
  // The session first, so that a rebuild replays the turns before this one and
  // not the message that is about to be sent.
  const currentSession = await createSessionWithTools();
  history.push({ role: 'user', content: message });
  let { text, toolCalls } = await streamResponse(currentSession, message);

  let rounds = 0;
  while (toolCalls.length) {
    if (++rounds > MAX_TOOL_CALLS) {
      log(`⚠️ Stopped after ${MAX_TOOL_CALLS} rounds of tool calls without an answer.\n`);
      return;
    }
    const toolResponses = [];
    for (const toolCall of toolCalls) {
      toolResponses.push({ type: 'tool-response', value: await runTool(tabId, toolCall) });
    }
    // Tool responses travel as a user message: the role enum only has
    // 'system', 'user' and 'assistant'.
    const messages = [{ role: 'user', content: toolResponses }];
    ({ text, toolCalls } = await streamResponse(session, messages));
  }

  if (text) history.push({ role: 'assistant', content: text });
}

// Prompts the on-device model without any tools, for one-off generations.
export async function promptOnDeviceModel(text) {
  const oneOffSession = await createSession();
  try {
    return await oneOffSession.prompt(text);
  } finally {
    oneOffSession.destroy();
  }
}

// Streams one model turn. The stream is heterogeneous: text arrives as plain
// strings, while each tool call arrives as its own `tool-call` chunk.
async function streamResponse(currentSession, input) {
  sidebar.trace.push({ userPrompt: input });
  const toolCalls = [];
  let text = '';

  for await (const chunk of currentSession.promptStreaming(input)) {
    if (typeof chunk !== 'string') {
      if (chunk?.type === 'tool-call') toolCalls.push(chunk.value);
      continue;
    }
    // Written straight to the results, so the answer appears as it is
    // generated rather than in one go when the turn ends.
    if (!text) sidebar.write('AI result: ');
    text += chunk;
    sidebar.write(chunk);
  }
  if (text) sidebar.write('\n\n');

  sidebar.trace.push({ response: { text, toolCalls: toolCalls.map(asPlainToolCall) } });
  if (!text && toolCalls.length === 0) log('⚠️ AI response has no text\n');
  return { text, toolCalls };
}

// Runs one tool the model asked for, and answers with a tool success or a tool
// error. Both are interfaces: a plain object is rejected.
async function runTool(tabId, toolCall) {
  const { callID, name } = toolCall;
  const tool = toolsByDeclaredName.get(name);
  const inputArgs = JSON.stringify(toolCall.arguments ?? {});

  if (!tool) {
    // The model made up a tool. Answering with a tool error rather than
    // throwing lets it correct itself on the next round.
    const errorMessage = `There is no tool named "${name}".`;
    log(`⚠️ ${errorMessage}`);
    sidebar.trace.push({ toolResponse: { callID, name, error: errorMessage } });
    return new LanguageModelToolError({ callID, name, errorMessage });
  }

  log(`AI calling tool "${tool.name}" with ${inputArgs}`);
  sidebar.trace.push({ toolCall: { callID, name: tool.name, inputArgs } });
  try {
    const result = await sidebar.executeTool(tabId, tool.name, inputArgs, tool.frameId);
    log(`Tool "${tool.name}" result: ${result}`);
    sidebar.trace.push({ toolResponse: { callID, name: tool.name, result } });

    return new LanguageModelToolSuccess({
      callID,
      name,
      // WebMCP tools answer with text. Chrome supports 'text' and 'object'
      // results, but not 'image' or 'audio'.
      result: [{ type: 'text', value: `${result ?? ''}` }],
    });
  } catch (e) {
    log(`⚠️ Error executing tool "${tool.name}": ${e.message}`);
    sidebar.trace.push({ toolResponse: { callID, name: tool.name, error: e.message } });
    return new LanguageModelToolError({ callID, name, errorMessage: e.message });
  }
}

// The tools the page has registered, as the model gets to see them. Tool names
// have to be unique, even when several frames register the same tool, so the
// mapping back to the tool that answers a call is kept alongside.
function getToolDeclarations() {
  toolsByDeclaredName = new Map();
  return (sidebar.getTools() || []).map((tool) => {
    let name = tool.name;
    while (toolsByDeclaredName.has(name)) name += '_';
    toolsByDeclaredName.set(name, tool);
    return {
      name,
      description: tool.description,
      inputSchema: tool.inputSchema
        ? JSON.parse(tool.inputSchema)
        : { type: 'object', properties: {} },
    };
  });
}

// Asking for the tool content types is what makes the model emit tool calls
// and accept their responses. A page without tools gets a plain session.
function getSessionOptions(declarations) {
  if (declarations.length === 0) return {};
  return {
    expectedInputs: [{ type: 'text' }, { type: 'tool-response' }],
    expectedOutputs: [{ type: 'text' }, { type: 'tool-call' }],
    tools: declarations,
  };
}

// Tools can only be declared when a session is created, so the session is
// rebuilt whenever the page's tools change. The conversation carries over: its
// text turns are replayed as initial prompts. Earlier tool calls are not, as
// the tools that answered them may be gone.
async function createSessionWithTools() {
  const declarations = getToolDeclarations();
  if (declarations.length > 0 && !isToolUseSupported()) {
    throw new Error(
      'Tool use is not enabled. Turn on chrome://flags/#prompt-api-tool-use to let the ' +
        'on-device model call the tools of this page.',
    );
  }

  const currentDeclarations = JSON.stringify(declarations);
  if (session && sessionDeclarations !== currentDeclarations) {
    log('The tools of this page changed. Starting a new on-device session with them.');
    session.destroy();
    session = undefined;
  }
  if (!session) {
    session = await createSession(getSessionOptions(declarations), {
      initialPrompts: [
        { role: 'system', content: sidebar.getSystemInstruction().join('\n') },
        ...history,
      ],
    });
    sessionDeclarations = currentDeclarations;
  }
  return session;
}

async function createSession(coreOptions = {}, options = {}) {
  if (!isPromptApiSupported()) {
    throw new Error(
      'The Prompt API is not available. See https://developer.chrome.com/docs/ai/get-started ' +
        'or pick a Gemini model instead.',
    );
  }
  // Asks about the very session that is about to be created, tools included.
  const availability = await LanguageModel.availability(coreOptions);
  if (availability === 'unavailable') {
    throw new Error('The on-device model is unavailable on this device.');
  }
  let lastProgress = -1;
  return await LanguageModel.create({
    ...coreOptions,
    ...options,
    monitor(monitor) {
      monitor.addEventListener('downloadprogress', ({ loaded, total }) => {
        const progress = Math.round((total ? loaded / total : loaded) * 100);
        if (progress === lastProgress) return;
        lastProgress = progress;
        log(`Downloading on-device model: ${progress}%`);
      });
    },
  });
}

// Tool calls keep their attributes on the prototype, so they need copying out
// by hand to end up in the trace as anything other than `{}`.
function asPlainToolCall({ callID, name, arguments: args }) {
  return { callID, name, arguments: args };
}

function log(text) {
  sidebar.write(`${text}\n`);
}
