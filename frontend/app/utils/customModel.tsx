
import { LLM, type BaseLLMParams } from "@langchain/core/language_models/llms";
import type { CallbackManagerForLLMRun } from "langchain/callbacks";
import { GenerationChunk } from "langchain/schema";

export interface CustomLLMInput extends BaseLLMParams {
  n: number;
}

export class CustomLLM extends LLM {
  n: number;

  constructor(fields: CustomLLMInput) {
    super(fields);
    this.n = fields.n;
  }

  _llmType() {
    return "custom";
  }

  async _call(
    prompt: string,
    options: this["ParsedCallOptions"],
    runManager: CallbackManagerForLLMRun
  ): Promise<string> {
    // Pass `runManager?.getChild()` when invoking internal runnables to enable tracing
    // await subRunnable.invoke(params, runManager?.getChild());
    return prompt.slice(0, this.n);
  }

  async *_streamResponseChunks(
    prompt: string,
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<GenerationChunk> {
    // Pass `runManager?.getChild()` when invoking internal runnables to enable tracing
    // await subRunnable.invoke(params, runManager?.getChild());
    for (const letter of prompt.slice(0, this.n)) {
      yield new GenerationChunk({
        text: letter,
      });
      // Trigger the appropriate callback
      await runManager?.handleLLMNewToken(letter);
    }
  }
}