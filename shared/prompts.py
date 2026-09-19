import random

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

# all texts are copied from authors.


def make_txt_record(
    key_patient: str,
    patient: dict,
    txts_note: list[str],
) -> str:
    txt_body = "\n".join(txts_note)
    return (
        f"<patient-record-{key_patient}>\n"
        f"Below is patient {patient['name']}'s medical record (ID: {key_patient}). \n"
        f"They were born on {patient['birthday']} and have the following diagnosis: {patient['diagnosis']}.\n"
        f"The patients medical record consists of {len(txts_note)} notes included below.\n"
        f"<notes>\n"
        f"{txt_body}\n"
        f"</notes>\n"
        f"</patient-record-{key_patient}>"
    )


def make_txt_ctx(
    key_patient: str,
    patient: dict,
    txt_note: str,
) -> str:
    # context template is from authors' hf dataset, not github.
    return (
        f"Below is a section of {patient['name']}'s medical record (ID: {key_patient}). \n"
        f"They were born on {patient['birthday']} and have the following diagnosis: {patient['diagnosis']}.\n"
        f"The patients medical record consists of {len(patient['texts'])} notes.\n"
        f"{txt_note}"
    )


def make_txt_inst() -> str:
    name_fmt = random.choice(["JSON", "YAML", "TOML", "INI", "XML", "plain text"])
    txts_inst = [
        [
            (
                f"Please generate a single chat message instructing an LLM to structure the information in {name_fmt}. "
                "Output only the chat message itself and absolutely nothing else. "
                "Make sure it is clear what section and document you are asking about. "
                "The message can follow the following template, filling in details from the corpus: \n\n"
                "'Can you structure the information in {{subsection}} of {{document}} related to {{something specific}} "
                f"in the following format: {name_fmt}? "
                "Be sure to include precise information like any dates, times, names, and numerical values.''"
            ),
            (
                f"Please generate a single chat message instructing an LLM to structure the information in {name_fmt}. "
                "Output only the chat message itself and absolutely nothing else. "
                "Make sure it is clear what section and document you are asking about. "
                "The message can follow the following template, filling in details from the corpus: \n\n"
                "'Can you structure the information in {{subsection}} of {{document}} "
                f"in the following format: {name_fmt}? "
                "Be sure to include precise information like any dates, times, names, and numerical values.''"
            ),
        ],
        [
            (
                "Please generate a single chat message instructing an LLM to summarize part of the corpus. "
                "Make sure the instruction is very explicit about the section of the corpus that you want to summarize. "
                "Include details (ids, names, titles, dates, etc.) that make it clear what you are asking about. "
            ),
            (
                "Please generate a single chat message instructing an LLM to summarize a section. "
                "Make sure the instruction is explicit about the section that should be summarized and the document it is from."
            ),
        ],
        [
            (
                "Generate a question for an LLM that will test its knowledge of the information in the corpus above. "
                "In your question be sure to include details (ids, names, titles, dates, etc.) that make it clear what you are asking about. "
                "Output only a single question. "
                "Do NOT include any other text or explanation other than the question."
            ),
            (
                "Generate a message for an LLM that will test its knowledge of the information in the corpus above."
                "Be sure to include details (ids, names, titles, dates, etc.) in the question so that it can be answered without access to the corpus (i.e. closed-book setting). "
                "Output only a single question. "
                "Do NOT include any other text or explanation other than the question."
            ),
            (
                "You are helping to quiz a user about the information in the corpus. "
                "Please generate a question about the subsection of the corpus above. "
                "Be sure to include details (ids, names, titles, dates, etc.) in the question to make it clear what you are asking about. "
                "Answer only with the question, do not include any other text."
            ),
        ],
        [
            (
                "You are working to train a language model on the information in the following corpus. "
                "Your primary goal is to think about practical, real-world tasks or applications that someone could achieve using the knowledge contained within this corpus. "
                "Consider how a user might want to apply this information, not just recall it. "
                "After considering potential use cases, your task will be to generate a sample question that reflects one of these downstream applications. "
                "This question/instruction/task should be something a user, who has access to this corpus, might ask when trying to accomplish their specific goal. "
                "Output only a single question. "
                "Do NOT include any other text or explanation other than the question."
            ),
        ],
        [
            (
                "You are having a creative conversation inspired by the information in the corpus. "
                "Please generate a question for your conversation partner to start off the discussion. "
                "Answer only with the question, do not include any other text."
            ),
        ],
    ]
    return random.choice(random.choice(txts_inst))


def make_txt_cot() -> str:
    txts_cot = [
        "Think before responding. Put your chain of thought/scratchpad between the <think> and </think> tags before providing your final response.",
        "If helpful, you can think before responding. Put your thinking between <thinking> and </thinking> tags. Then, provide your final response between <response> and </response> tags.",
        "Respond in the following format: <thinking>...</thinking> <response>...</response>",
        "Explain your reasoning before providing your final response.",
        "Explain your reasonining between <reasoning> and </reasoning> tags.",
        "Provide your final answer within <answer>...</answer> tags. Optionally, you can explain your reasoning between <reasoning> and </reasoning> tags.",
        "You may include your reasoning before answering. Use <reasoning>...</reasoning> to enclose your thoughts, and <final>...</final> for your answer.",
        "First, think through the problem and enclose your thoughts in <thought>...</thought>. Then, present your answer clearly in <output>...</output>.",
        "Use <step-by-step>...</step-by-step> for your intermediate reasoning, followed by <answer>...</answer> for the final result.",
        "Start with your analysis in <deliberation>...</deliberation>. Conclude with a clear answer in <response>...</response>.",
        "You may show your chain of thought in <chain>...</chain> and state your final decision in <decision>...</decision>.",
        "Wrap your reasoning process in <logic>...</logic>, and your ultimate conclusion in <conclusion>...</conclusion>.",
        "Think carefully before answering. Put your process in <process>...</process> and your solution in <solution>...</solution>.",
        "Please provide your thought process first, enclosed in <rationale>...</rationale>, and then give your final answer in <final_answer>...</final_answer>.",
        "Include your reasoning in <analysis>...</analysis> and your conclusion in <result>...</result>.",
        "Begin with <thinking_process>...</thinking_process> to show how you reasoned through the problem. Finish with <response>...</response>.",
        "Use <explanation>...</explanation> to walk through the logic. Then state your answer in <output>...</output>.",
        "Present your logical steps in <reasoning_chain>...</reasoning_chain> and conclude with <final_response>...</final_response>.",
        "Start with <evaluation>...</evaluation> to explain how you analyzed the question. Then, give your answer in <decision>...</decision>.",
        "Outline your reasoning in <deduction>...</deduction> and present the final answer in <resolution>...</resolution>.",
        "First explain in <justification>...</justification>, then give your definitive answer in <answer>...</answer>.",
        "Place your step-by-step logic in <path>...</path> and your outcome in <solution>...</solution>.",
        "Break down the problem in <walkthrough>...</walkthrough> before stating your answer in <conclusion>...</conclusion>.",
        "Reason through the problem in <examine>...</examine> and finalize with <respond>...</respond>.",
        "Give your thought process inside <trace>...</trace> and the answer inside <reply>...</reply>.",
        "Write your full reasoning under <work>...</work>, then state the answer clearly in <end>...</end>.",
        "Use <rundown>...</rundown> to explain your steps, and <final>...</final> to share your answer.",
        "First, walk through your reasoning process step by step. Then, clearly state your final answer.",
        "Begin by explaining how you approach the problem. Afterward, give your final response.",
        "Start with a detailed breakdown of your thought process. Conclude with a concise answer.",
        "Explain your logic as you work through the problem. When you're done, provide your conclusion.",
        "Think out loud as you reason through the question. End with a definitive answer.",
        "Work through the problem in detail, reasoning carefully. Then summarize your final decision.",
        "Describe each step you take to solve the problem. Finish by stating the final result.",
        "Provide a thorough explanation of how you arrive at your answer. Then state the answer clearly.",
        "Show your reasoning process from start to finish. Make sure to give your final answer at the end.",
        "Break the problem into logical steps and explain each one. Then give your final response.",
        "Write out your reasoning clearly and methodically. Conclude with your final conclusion.",
        "Reflect on the problem and describe your full reasoning. Then, say what your answer is.",
        "Think critically about the question and narrate your process. Then provide your final decision.",
        "Show your internal reasoning inside `{{Rationale}}...{{/Rationale}}`. Give the final conclusion inside `{{Conclusion}}...{{/Conclusion}}`.",
        "Detail your problem-solving steps between `[Steps]...[/Steps]`. Provide the answer between `[Result]...[/Result]`.",
        "Explain the derivation process using `Derivation: ... End Derivation`. State the final output using `Output: ... End Output`.",
        "Map out your thought path within `<Path>...</Path>`. State the final destination within `<Destination>...</Destination>`.",
        "Provide your analysis enclosed in `<<Analysis>>...<</Analysis>>`. Present the determined answer enclosed in `<<Answer>>...<</Answer>>`.",
        "Elaborate on your thinking process with `// Elaboration Start` and `// Elaboration End`. Provide the final response with `// Response Start` and `// Response End`.",
        "Use `(Thought Process: ... )` to show your thinking. Use `(Final Answer: ... )` for the result.",
        "Lay out the groundwork in `<Foundation>...</Foundation>`. Build the final answer in `<Structure>...</Structure>`.",
        "Document the logical flow in `{* Logic Flow *} ... {* /Logic Flow *}`. Deliver the outcome in `{* Outcome *} ... {* /Outcome *}`.",
        "Begin with your deliberation, marked by `Deliberation: ...`. Conclude with your final decision, marked by `Decision: ...`.",
        "Record your internal monologue in `<Monologue>...</Monologue>`. State the external response in `<Statement>...</Statement>`.",
        "Chart the sequence of reasoning within `[[Sequence]]...[[/Sequence]]`. Present the end point within `[[Endpoint]]...[[/Endpoint]]`.",
        "Dissect the problem in `<Dissection>...</Dissection>`. Synthesize the answer in `<Synthesis>...</Synthesis>`.",
        "Narrate your thought process using `Narrative: ...`. Provide the concluding answer using `Conclusion: ...`.",
        "Outline your strategy in `<Strategy>...</Strategy>`. Execute the final answer in `<Execution>...</Execution>`.",
        "Feel free to think it through out loud first, then just drop your answer at the end.",
        "Walk yourself through the problem—no rush. Once you're set, say what you'd go with.",
        "You can talk it out step by step. Just wrap up with whatever you think the answer is.",
        "Think it through however you like, and then let me know your final call.",
        "Start by working through the logic in your own way. When you're done, give the answer.",
        "Lay out your thinking as it comes to you. At the end, just say what you'd choose.",
        "Break it down how you want, no pressure. Then tell me what your final answer would be.",
        "Talk yourself through the reasoning part. When it feels right, give your answer.",
        "Explain it like you're figuring it out in real time, then land on your pick.",
        "Take a moment to think it through, and when you're ready, just say your answer.",
        "Work it out however makes sense to you. Then drop your answer when you're good.",
        "Go step by step, like you're thinking out loud. End with whatever answer you'd settle on.",
        "Walk through your reasoning step by step, and once it all clicks, share your answer.",
        "Take me through your thought process from start to finish, then tell me your conclusion.",
        "Think out loud as you piece it together. When you’re ready, just state your answer.",
        "Break down the problem in your own words, then wrap up with your choice.",
        "Talk through each part as you solve it, and then give your final answer.",
        "Map out your logic as you go, and once you’re confident, give your answer.",
        "Reason it out at your own pace, and when you’re set, let me know your answer.",
        "Feel free to puzzle it out stepwise. At the end, just say your final pick.",
        "Process the question however you like. Once you’ve worked it out, share your answer.",
        "Unpack the problem in detail, and when you’re ready, provide your answer.",
        "Work through the details in your own style, and then land on your answer.",
        "Think through the scenario, explaining as you go, and finish with your answer.",
        "Go through your reasoning openly, and then make your final call.",
        "Lay out your analysis clearly, and when you reach a conclusion, share it.",
        "Step through your logic in real time, and end with your decision.",
        "Explain your approach as you solve it, then just give your answer at the end.",
        "Go through the motions of solving it, and when you’re done, state your answer.",
        "Work the problem out in your own way, and when you’re ready, say your answer.",
        "Detail your thought process as it unfolds, then close with your answer.",
        "Take your time to reason through it, and when you’ve got it, give your answer.",
        "Unpack your reasoning in <logic>...</logic>. Conclude with your solution in <result>...</result>.",
        "Share your thought process inside [Reasoning]...[/Reasoning]. State your answer in [Answer]...[/Answer].",
        "Walk through your approach in <<<Thinking>>>...<<<End Thinking>>>. Provide your answer in <<<Answer>>>...<<<End Answer>>>.",
        "Place your analysis in <Breakdown>...</Breakdown>. Offer the answer in <Solution>...</Solution>.",
        "Use [[Rationale]]...[[/Rationale]] for your reasoning. Use [[Conclusion]]...[[/Conclusion]] for your answer.",
        "Think through the problem in # Reasoning: ... # End Reasoning. Give your answer in # Answer: ... # End Answer.",
        "Lay out your process in <Process>...</Process>. Deliver your conclusion in <Conclusion>...</Conclusion>.",
        "Step through your logic in {Reasoning: ... }. Present your answer in {Answer: ... }.",
        "Present your thought process in --- Reasoning --- ... --- End Reasoning ---. State your answer in --- Answer --- ... --- End Answer ---.",
        "Outline your thinking in (Reasoning Start)...(Reasoning End). Wrap up with (Answer Start)...(Answer End).",
        "Describe your approach using <Approach>...</Approach>. Summarize your answer using <Summary>...</Summary>.",
        "Go through your logic in [Logic Path]...[/Logic Path]. Finalize with [Final Response]...[/Final Response].",
        "Map out your steps in <<Steps>>...<</Steps>>. Place your answer in <<Result>>...<</Result>>.",
        "Narrate your reasoning in <Explanation>...</Explanation>. State your answer in <Reply>...</Reply>.",
        "Explain your process inside {Process: ...}. Conclude with {Result: ...}.",
        "Break down your logic in [Analysis]...[/Analysis]. Conclude with your answer in [Conclusion]...[/Conclusion].",
        "Think out your process in <Thoughts>...</Thoughts>. Share your answer in <Answer>...</Answer>.",
        "Detail your reasoning in [[Analysis]]...[[/Analysis]]. Give the answer in [[Final]]...[[/Final]].",
        "Work through your thoughts in --- Process --- ... --- /Process ---. Finish with --- Solution --- ... --- /Solution ---.",
        "Talk through your logic in {Deliberation}...{/Deliberation}. Give your answer in {Decision}...{/Decision}.",
    ]
    return random.choice(txts_cot)


def make_txt_sys(
    txt_ctx: str,
) -> str:
    return f"\nYou are in a conversation about the following user information.\n\n<info>\n{txt_ctx}\n</info>"


def make_msg_sys(
    txt: str,
) -> dict:
    return {"role": "system", "content": txt}


def make_msg_user(
    txt: str,
) -> dict:
    return {"role": "user", "content": txt}


def make_id(
    tokenizer: PreTrainedTokenizerBase,
    msgs: list[dict],
    type_model: str,
    device: torch.device,
) -> Tensor:
    kwargs_tpl = {}
    match type_model:
        case "llama":
            # fix date in llama system header not to be affected by run date.
            kwargs_tpl["date_string"] = "26 Jul 2024"
        case _:
            raise NotImplementedError(type_model)
    return tokenizer.apply_chat_template(msgs, return_tensors="pt", return_dict=False, **kwargs_tpl).to(device)


def make_id_inp(
    tokenizer: PreTrainedTokenizerBase,
    msgs: list[list[dict]],
    type_model: str,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    kwargs_tpl = {}
    match type_model:
        case "llama":
            # fix date in llama system header not to be affected by run date.
            kwargs_tpl["date_string"] = "26 Jul 2024"
        case _:
            raise NotImplementedError(type_model)
    enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, padding=True, return_tensors="pt", return_dict=True, **kwargs_tpl)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)
