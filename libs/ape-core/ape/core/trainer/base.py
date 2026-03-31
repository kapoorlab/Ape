import asyncio
import copy
import inspect
import json
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from ape.common.generator import BaseGenerator
from ape.common.global_metric import BaseGlobalMetric
from ape.common.metric import BaseMetric
from ape.common.prompt import Prompt
from ape.common.types import GlobalMetricResult, MetricResult, DatasetItem
from ape.common.utils.logging import logger
from ape.core.core_prompts import ApeCorePrompts
from ape.core.types.report import BaseReport
from ape.core.utils import extract_prompt


class BaseTrainer(ABC):
    def __init__(
        self,
        generator: BaseGenerator,
        metric: BaseMetric,
        global_metric: Optional[BaseGlobalMetric] = None,
        task_description: Optional[str] = None,
        metric_description: Optional[str] = None,
        testmode: Optional[bool] = False,
        optimizer_model: Optional[str] = None,
        optimizer_response_format: Optional[dict] = None,
        **kwargs,
    ):
        self.generate = generator
        self.metric = metric
        self.global_metric = global_metric
        self.task_description = task_description
        self.metric_description = metric_description
        self.dataset_summary = None
        self.testmode = testmode
        self.optimizer_model = optimizer_model
        self.optimizer_response_format = optimizer_response_format

    def _override_prompt_model(self, prompt: "Prompt") -> "Prompt":
        """Override the model/response_format on an Ape Prompt if optimizer_model is set."""
        if self.optimizer_model is not None:
            prompt.model = self.optimizer_model
        if self.optimizer_response_format is not None:
            prompt.response_format = self.optimizer_response_format
        return prompt

    @abstractmethod
    async def train(
        self,
        prompt: Prompt,
        trainset: List[DatasetItem],
        valset: List[DatasetItem],
    ) -> Tuple[Prompt, BaseReport]:
        """
        Train the prompt

        Args:
            prompt (Prompt): Prompt
            trainset (List[DatasetItem]): Training dataset
            valset (List[DatasetItem]): Validation dataset

        Returns:
            Tuple[Prompt, BaseReport]: Trained prompt and report
        """
        pass

    async def __call__(
        self, prompt: Prompt, trainset: List[DatasetItem], valset: List[DatasetItem]
    ) -> Tuple[Prompt, BaseReport]:
        return await self.train(prompt=prompt, trainset=trainset, valset=valset)

    async def _evaluate(
        self, dataset: List[DatasetItem], prompt: Prompt
    ) -> Tuple[List[Any], List[MetricResult], GlobalMetricResult]:
        """
        Evaluate a dataset using the generator and metric.

        Args:
            dataset (List[DatasetItem]): The dataset to evaluate.
            prompt (Prompt): The prompt to use for generation.

        Returns:
            GlobalMetricResult: The aggregated metric result for the dataset.
        """
        # Asynchronously generate predictions and compute metrics for each item
        generate_tasks = [
            self.generate(
                prompt=prompt,
                inputs=item["inputs"],
            )
            for item in dataset
        ]
        preds = []
        for i in range(0, len(generate_tasks), 50):
            preds.extend(await asyncio.gather(*generate_tasks[i:i+50]))

        metric_tasks = [
            self.metric(
                dataset_item=item,
                pred=pred,
            )
            for item, pred in zip(dataset, preds)
        ]
        eval_results = []
        for i in range(0, len(metric_tasks), 50):
            eval_results.extend(await asyncio.gather(*metric_tasks[i:i+50]))

        # Compute the global metric
        global_score = await self.global_metric(eval_results)
        return preds, eval_results, global_score

    async def _generate_task_description(
        self,
        prompt: Prompt,
        trainset: List[DatasetItem],
    ) -> str:
        describe_prompt = ApeCorePrompts.get("describe-prompt")
        self._override_prompt_model(describe_prompt)

        temperature = describe_prompt.temperature

        for attempt in range(3):
            try:
                # Generate dataset summary
                dataset_summary = await self._dataset_summarizer(trainset)
                self.dataset_summary = dataset_summary

                # Describe the prompt
                prompt_description = await describe_prompt(
                    lm_config=dict(temperature=temperature),
                    prompt=str(prompt.messages),
                    dataset_description=dataset_summary,
                )

                prompt_description = prompt_description["description"]

                return prompt_description
            except Exception as e:
                if attempt == 2:  # Last attempt
                    logger.exception("Error generating task description")
                    return ""
                logger.warning(
                    f"Error generating task description: {e}. Retrying... (Attempt {attempt + 1}/3)"
                )
                await asyncio.sleep(1)  # Wait for 1 second before retrying
                temperature += 0.1

    async def _generate_metric_description(
        self,
    ) -> str:

        gen_metric_description: Prompt = ApeCorePrompts.get("gen-metric-description")
        self._override_prompt_model(gen_metric_description)
        gen_metric_description_with_global_metric: Prompt = ApeCorePrompts.get(
            "gen-metric-description-with-global-metric"
        )
        self._override_prompt_model(gen_metric_description_with_global_metric)

        compute_function = getattr(self.metric, "compute", None)
        compute_function_source_code = inspect.getsource(compute_function)

        if self.global_metric:
            global_metric_compute_function = getattr(self.global_metric, "compute", None)
            global_metric_compute_function_source_code = inspect.getsource(
                global_metric_compute_function
            )

            # get Prompt gen-metric-description-with-global-metric.prompt
            metric_str = await gen_metric_description_with_global_metric(
                **{
                    "metric_sourcecode": compute_function_source_code,
                    "global_metric_sourcecode": global_metric_compute_function_source_code,
                }
            )

        else:
            metric_str = await gen_metric_description(
                **{
                    "metric_sourcecode": compute_function_source_code,
                }
            )
        return metric_str

    async def _dataset_summarizer(
        self,
        trainset: List[DatasetItem],
        view_data_batch_size: Optional[int] = 10,
    ) -> str:
        upper_lim = min(len(trainset), view_data_batch_size)

        descriptor = ApeCorePrompts.get("dataset-descriptor")
        self._override_prompt_model(descriptor)
        temperature = 0.7

        for attempt in range(3):
            try:
                formatted_examples = self._format_examples(trainset, upper_lim)

                output = await descriptor(
                    lm_config=dict(temperature=temperature),
                    examples=formatted_examples,
                )
                res = ""
                for line in output["observations"]:
                    res += line + "\n"
                return res
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise e
                logger.warning(
                    f"Error summarizing dataset: {e}. Retrying... (Attempt {attempt + 1}/3)"
                )
                await asyncio.sleep(1)  # Wait for 1 second before retrying
                temperature += 0.1

    def _format_examples(self, trainset: List[DatasetItem], batch_size: int) -> str:
        formatted_examples = ""
        random_samples = random.sample(trainset, min(batch_size, len(trainset)))

        for idx, item in enumerate(random_samples):
            formatted_examples += f"### Demo {idx+1} ###\n"

            inputs, outputs = item["inputs"], item["outputs"]

            formatted_examples += "**Inputs**\n"
            for key, value in inputs.items():
                formatted_examples += f"{key.capitalize()}:\n{value}\n"

            formatted_examples += f"**Outputs**\n{json.dumps(outputs, indent=2)}\n\n"

        return formatted_examples

    def _fallback_fewshot_placeholder(self, prompt: Prompt) -> Prompt:
        """Manually inject {_FEWSHOT_} placeholder when LLM fails to produce valid JSON."""
        new_prompt = copy.deepcopy(prompt)
        placeholder = "\nExample:\n{_FEWSHOT_}"
        for msg in new_prompt.messages:
            if msg["role"] == "system":
                msg["content"] += placeholder
                return new_prompt
        # No system message — append to last user message
        if new_prompt.messages:
            new_prompt.messages[-1]["content"] += placeholder
        return new_prompt

    @staticmethod
    def _extract_json(raw):
        """Try to extract a JSON object from a raw LLM response (str or dict)."""
        if isinstance(raw, dict):
            if "messages" in raw:
                return raw
            # Some providers nest under "output" or similar
            for key in ("output", "result", "response"):
                if key in raw and isinstance(raw[key], dict) and "messages" in raw[key]:
                    return raw[key]
        if isinstance(raw, str):
            # Strip thinking tags if present
            text = raw
            if "<think>" in text:
                idx = text.rfind("</think>")
                if idx != -1:
                    text = text[idx + len("</think>"):]
            text = text.strip()
            # Try JSON array → wrap as messages
            if text.startswith("["):
                end = text.rfind("]") + 1
                if end > 0:
                    try:
                        arr = json.loads(text[:end])
                        if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], dict) and "role" in arr[0]:
                            return {"messages": arr}
                    except (json.JSONDecodeError, ValueError):
                        pass
            # Find JSON object
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    obj = json.loads(text[start:end])
                    if isinstance(obj, dict):
                        return obj
                except (json.JSONDecodeError, ValueError):
                    pass
                # Try ast.literal_eval for Python dict literals
                import ast
                try:
                    obj = ast.literal_eval(text[start:end])
                    if isinstance(obj, dict):
                        return obj
                except (ValueError, SyntaxError):
                    pass
        return None

    @staticmethod
    def _extract_messages(output):
        """Extract messages list from a raw LLM output that may not follow the expected schema.

        Handles cases where Ollama returns:
        - {"messages": [...]}  (expected format)
        - {"response": "{'system': '...', 'user': '...'}"}  (stringified Python dict)
        - {"system": "content...", "user": "content..."}  (role-keyed with string values)
        - {"system": {"content": ..., "role": ...}, "user": {...}}  (role-keyed with dict values)
        Returns a list of message dicts [{"role": ..., "content": ...}] or None.
        """
        if output is None or (isinstance(output, dict) and not output):
            return None

        # Direct messages key
        if isinstance(output, dict) and "messages" in output:
            msgs = output["messages"]
            if isinstance(msgs, list) and len(msgs) > 0:
                return msgs

        # Try to find messages in string values (model sometimes stringifies dicts in "response" key)
        if isinstance(output, dict):
            for key in ("response", "output", "result", "prompt"):
                val = output.get(key)
                if isinstance(val, str):
                    parsed = BaseTrainer._try_parse_dict(val)
                    if parsed:
                        if "messages" in parsed:
                            return parsed["messages"]
                        msgs = BaseTrainer._dict_roles_to_messages(parsed)
                        if msgs:
                            return msgs

            # Direct role-keyed dict
            msgs = BaseTrainer._dict_roles_to_messages(output)
            if msgs:
                return msgs

        return None

    @staticmethod
    def _try_parse_dict(text):
        """Try to parse a string as JSON or Python dict literal."""
        import ast
        # Try JSON first
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
        # Try Python literal (handles single quotes, escaped newlines)
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                return obj
        except (ValueError, SyntaxError):
            pass
        # Try finding a dict inside the string
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            substr = text[start:end]
            try:
                obj = json.loads(substr)
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass
            try:
                obj = ast.literal_eval(substr)
                if isinstance(obj, dict):
                    return obj
            except (ValueError, SyntaxError):
                pass
        return None

    @staticmethod
    def _dict_roles_to_messages(d):
        """Convert role-keyed dicts to messages list.

        Handles both:
        - {"system": {"content": "...", "role": "system"}, "user": {...}}
        - {"system": "content string", "user": "content string"}
        """
        msgs = []
        for role in ("system", "user", "assistant"):
            if role in d:
                val = d[role]
                if isinstance(val, dict) and "content" in val:
                    msgs.append({"role": role, "content": val["content"]})
                elif isinstance(val, str) and len(val) > 5:
                    msgs.append({"role": role, "content": val})
        return msgs if len(msgs) >= 1 else None

    def _extract_prompt_messages(self, output, base_prompt: "Prompt"):
        """Aggressively extract messages from any LLM output format.

        Tries structured extraction first, then falls back to treating
        any long string value as a new system prompt.
        Raises KeyError if nothing useful can be extracted.
        """
        # 1. Direct messages key
        if isinstance(output, dict) and "messages" in output:
            msgs = output["messages"]
            if isinstance(msgs, list) and len(msgs) > 0:
                return msgs

        # 2. Robust extraction (handles role-keyed dicts, stringified dicts, etc.)
        msgs = self._extract_messages(output)
        if msgs:
            return msgs

        # 3. JSON/ast extraction
        parsed = self._extract_json(output)
        if parsed and "messages" in parsed:
            return parsed["messages"]
        if parsed:
            msgs = self._extract_messages(parsed)
            if msgs:
                return msgs

        # 4. Last resort: find the longest string value in the output and use it
        #    as the system prompt content (qwen3 often returns arbitrary keys with
        #    the actual prompt text as a value)
        if isinstance(output, dict):
            best_text = ""
            for val in output.values():
                text = val if isinstance(val, str) else ""
                # Try to parse stringified dicts/lists inside values
                if isinstance(val, str) and len(val) > 20:
                    inner = self._extract_json(val)
                    if inner:
                        if "messages" in inner:
                            return inner["messages"]
                        inner_msgs = self._extract_messages(inner)
                        if inner_msgs:
                            return inner_msgs
                if len(text) > len(best_text):
                    best_text = text
            if len(best_text) > 20:
                # Build messages mirroring base_prompt structure but with new content
                new_messages = []
                for msg in base_prompt.messages:
                    if msg["role"] == "system":
                        new_messages.append({"role": "system", "content": best_text})
                    else:
                        new_messages.append(dict(msg))
                if new_messages:
                    return new_messages

        if isinstance(output, str) and len(output) > 20:
            # Raw string — use as system prompt
            text = output
            if "<think>" in text:
                idx = text.rfind("</think>")
                if idx != -1:
                    text = text[idx + len("</think>"):].strip()
            if len(text) > 20:
                new_messages = []
                for msg in base_prompt.messages:
                    if msg["role"] == "system":
                        new_messages.append({"role": "system", "content": text})
                    else:
                        new_messages.append(dict(msg))
                if new_messages:
                    return new_messages

        keys = list(output.keys()) if isinstance(output, dict) else type(output).__name__
        raise KeyError(f"Could not extract messages from output ({keys})")

    async def generate_fewshot_placeholder(self, prompt: Prompt) -> Prompt:
        fewshot_placeholder_generator = ApeCorePrompts.get("gen-fewshot-placeholder")
        self._override_prompt_model(fewshot_placeholder_generator)

        try:
            new_prompt_raw = await fewshot_placeholder_generator(prompt=str(prompt.messages), _retry_count=0)

            messages = None
            if isinstance(new_prompt_raw, dict) and "messages" in new_prompt_raw:
                messages = new_prompt_raw["messages"]
            if messages is None:
                parsed = self._extract_json(new_prompt_raw)
                if parsed and "messages" in parsed:
                    messages = parsed["messages"]
            if messages is None:
                messages = self._extract_messages(new_prompt_raw)

            if messages is not None:
                new_prompt = copy.deepcopy(prompt)
                new_prompt.messages = messages
                return new_prompt
        except Exception:
            pass

        logger.warning("Fewshot placeholder LLM failed, using direct injection.")
        return self._fallback_fewshot_placeholder(prompt)
