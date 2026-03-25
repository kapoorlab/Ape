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
            # Find JSON object
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        return None

    @staticmethod
    def _extract_messages(output):
        """Extract messages list from a raw LLM output that may not follow the expected schema.

        Handles cases where Ollama returns:
        - {"messages": [...]}  (expected format)
        - {"response": '{"system": {...}, "user": {...}}'}  (stringified nested JSON)
        - {"system": {"content": ..., "role": ...}, "user": {"content": ..., "role": ...}}
        - Any dict with "role" and "content" values buried somewhere
        Returns a list of message dicts [{"role": ..., "content": ...}] or None.
        """
        if output is None or (isinstance(output, dict) and not output):
            return None

        # Direct messages key
        if isinstance(output, dict) and "messages" in output:
            msgs = output["messages"]
            if isinstance(msgs, list) and len(msgs) > 0:
                return msgs

        # Try to find messages in string values (model sometimes stringifies JSON in "response" key)
        if isinstance(output, dict):
            for key in ("response", "output", "result", "prompt"):
                val = output.get(key)
                if isinstance(val, str):
                    try:
                        parsed = json.loads(val.replace("'", '"'))
                        if isinstance(parsed, dict):
                            if "messages" in parsed:
                                return parsed["messages"]
                            # {"system": {"content":..,"role":..}, "user": {"content":..,"role":..}}
                            msgs = BaseTrainer._dict_roles_to_messages(parsed)
                            if msgs:
                                return msgs
                    except (json.JSONDecodeError, ValueError):
                        pass

            # Direct role-keyed dict: {"system": {"content": ..}, "user": {"content": ..}}
            msgs = BaseTrainer._dict_roles_to_messages(output)
            if msgs:
                return msgs

        return None

    @staticmethod
    def _dict_roles_to_messages(d):
        """Convert {"system": {"content":..,"role":..}, "user": {"content":..,"role":..}} to messages list."""
        msgs = []
        for role in ("system", "user", "assistant"):
            if role in d and isinstance(d[role], dict) and "content" in d[role]:
                msgs.append({"role": role, "content": d[role]["content"]})
        return msgs if len(msgs) >= 1 else None

    async def generate_fewshot_placeholder(self, prompt: Prompt) -> Prompt:
        fewshot_placeholder_generator = ApeCorePrompts.get("gen-fewshot-placeholder")
        self._override_prompt_model(fewshot_placeholder_generator)

        retry_count = 0
        while retry_count < 5:
            try:
                new_prompt_raw = await fewshot_placeholder_generator(prompt=str(prompt.messages), _retry_count=retry_count)
                parsed = self._extract_json(new_prompt_raw)
                if parsed is None or "messages" not in parsed:
                    raise KeyError("messages")
                new_prompt = copy.deepcopy(prompt)
                new_prompt.messages = parsed["messages"]
                return new_prompt
            except Exception as exc:
                logger.warning(
                    f"Error generating fewshot placeholder: {exc}. Retrying... (Attempt {retry_count + 1}/5)"
                )
                retry_count += 1
                if retry_count == 5:
                    logger.warning("All attempts failed. Using fallback fewshot placeholder injection.")
                    return self._fallback_fewshot_placeholder(prompt)
