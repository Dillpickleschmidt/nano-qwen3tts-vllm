import asyncio
import queue
import threading
import uuid
import torch
import soundfile as sf
import time

from nano_qwen3tts_vllm.utils.prompt import prepare_custom_voice_prompt
from nano_qwen3tts_vllm.utils.voice_clone import load_voice_prompt, prepare_speaker_embeds
from nano_qwen3tts_vllm.processor import Qwen3TTSProcessor
from nano_qwen3tts_vllm.utils.generation import prepare_inputs
from nano_qwen3tts_vllm.llm import TalkerLLM, PredictorLLM
from nano_qwen3tts_vllm.sampling_params import SamplingParams


torch.manual_seed(42)
# Use Qwen3-TTS processor when available so tokenization matches Qwen3TTSModel.generate_custom_voice exactly
def _get_processor(model_path: str):
    try:
        from qwen_tts.core.models import Qwen3TTSProcessor as Qwen3TTSProcessorHF
        return Qwen3TTSProcessorHF.from_pretrained(model_path, fix_mistral_regex=True)
    except ImportError:
        return Qwen3TTSProcessor.from_pretrained(model_path, fix_mistral_regex=True)


class Qwen3TTSInterface:
    def __init__(self, model_path: str, enforce_eager: bool = False, tensor_parallel_size: int = 1, zmq_bridge=None):
        self.model_path = model_path
        self.enforce_eager = enforce_eager
        self.tensor_parallel_size = tensor_parallel_size
        self.zmq_bridge = zmq_bridge
        self.talker_llm = TalkerLLM(model_path, enforce_eager=enforce_eager, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.3)
        self.predictor_llm = PredictorLLM(model_path, enforce_eager=enforce_eager, tensor_parallel_size=tensor_parallel_size)
        self.processor = _get_processor(model_path)
        self.model_config = self.talker_llm.model_runner.full_config
        
        self.text_embedding = self.talker_llm.model_runner.model.get_text_embeddings()
        self.input_embedding = self.talker_llm.model_runner.model.get_input_embeddings()
        self.text_projection = self.talker_llm.model_runner.model.text_projection
        
        self.predictor_input_embeddings = self.predictor_llm.model_runner.model.model.codec_embedding
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ZMQ path: asyncio queues; dispatcher and engine run as asyncio tasks (no threads).
        self._request_queues: dict[str, asyncio.Queue] = {}
        self._queues_lock = asyncio.Lock()
        self._zmq_tasks: list[asyncio.Task] = []
        self._zmq_inbox: queue.Queue | None = None
        self._zmq_tasks_started = False
        # Serialize request prep (GPU work) so event loop can run engine while another request prepares.
        self._prep_lock = threading.Lock()

    async def start_zmq_tasks(self) -> None:
        """Start the ZMQ dispatcher (thread + asyncio task) and engine loop. Call once before generate_async when zmq_bridge is set."""
        if self.zmq_bridge is None:
            raise RuntimeError("start_zmq_tasks requires zmq_bridge to be set on the interface")
        if self._zmq_tasks_started:
            return
        self._zmq_tasks_started = True
        from nano_qwen3tts_vllm.zmq.dispatcher import start_dispatcher_thread, run_dispatch_loop
        from nano_qwen3tts_vllm.zmq.engine_loop import run_engine_loop
        # Dedicated thread for blocking ZMQ recv (so it is already waiting before first publish)
        _, inbox = start_dispatcher_thread(self.zmq_bridge.bind_address)
        self._zmq_inbox = inbox
        t1 = asyncio.create_task(run_dispatch_loop(inbox, self._request_queues, self._queues_lock))
        t2 = asyncio.create_task(run_engine_loop(
            self.talker_llm, self.predictor_llm, self.zmq_bridge
        ))
        self._zmq_tasks.extend([t1, t2])
        # Give the recv thread time to connect and enter recv (avoid ZMQ slow-joiner)
        await asyncio.sleep(0.2)

    async def stop_zmq_tasks(self) -> None:
        """Stop ZMQ tasks so the event loop can exit. Puts sentinel in inbox to unblock executor thread."""
        if not self._zmq_tasks:
            return
        # Unblock the thread blocked on inbox.get() in run_in_executor so shutdown_default_executor() can finish
        if self._zmq_inbox is not None:
            self._zmq_inbox.put(None)
        for t in self._zmq_tasks:
            t.cancel()
        await asyncio.gather(*self._zmq_tasks, return_exceptions=True)
        self._zmq_tasks.clear()
        self._zmq_inbox = None

    def generate_custom_voice(self, text: str, language: str = "English", speaker: str = "Vivian"):
        """Sync generator. Only valid when zmq_bridge is None. For ZMQ use generate_custom_voice_async()."""
        if self.zmq_bridge is not None:
            raise RuntimeError(
                "When using ZMQ bridge, use async API: await interface.start_zmq_tasks(); "
                "async for chunk in interface.generate_custom_voice_async(...)"
            )
        input_ids, instruct_ids, speakers, languages = prepare_custom_voice_prompt(
            text=text, language=language, speaker=speaker,
            processor=self.processor, device=self.device,
        )
        talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask = prepare_inputs(
            config=self.model_config,
            input_ids=input_ids, instruct_ids=instruct_ids, speakers=speakers, languages=languages,
            non_streaming_mode=True,
            text_embedding=self.text_embedding, input_embedding=self.input_embedding,
            text_projection=self.text_projection, device=self.device,
        )
        yield from self._generate_caller_driven(
            talker_input_embeds, trailing_text_hiddens, tts_pad_embed,
            str(uuid.uuid4()),
            SamplingParams(temperature=1.0, max_tokens=1),
            SamplingParams(temperature=0.9, max_tokens=17),
        )

    def generate_voice_clone(
        self,
        text: str,
        voice_clone_prompt: dict,
        language: str = "english",
        temperature: float = 0.9,
        top_p: float = 1.0,
    ):
        """Generate speech using a pre-loaded voice clone prompt.

        Args:
            text: Text to synthesize
            voice_clone_prompt: Dict from load_voice_prompt() with keys:
                ref_code, ref_spk_embedding, x_vector_only_mode, icl_mode, ref_text
            language: Language code (e.g., "english", "chinese", "auto")
            temperature: Sampling temperature for predictor
            top_p: Top-p sampling for predictor

        Yields:
            List of 16 codebook IDs per generation step
        """
        if self.zmq_bridge is not None:
            raise RuntimeError("generate_voice_clone requires zmq_bridge=None; use sync mode")

        input_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = [self.processor(text=input_text, return_tensors="pt").input_ids.to(self.device)]

        ref_ids = None
        is_icl_mode = voice_clone_prompt["icl_mode"][0]
        ref_code = voice_clone_prompt["ref_code"][0] if voice_clone_prompt.get("ref_code") else None

        if ref_code is not None and is_icl_mode:
            ref_text = voice_clone_prompt["ref_text"][0]
            if ref_text:
                ref_input_text = f"<|im_start|>assistant\n{ref_text}<|im_end|>"
                ref_ids = [self.processor(text=ref_input_text, return_tensors="pt").input_ids.to(self.device)]

        voice_clone_spk_embeds = prepare_speaker_embeds(
            voice_clone_prompt, self.device, torch.bfloat16
        )

        generate_icl_prompt_fn = None
        if ref_code is not None and is_icl_mode:
            generate_icl_prompt_fn = self._make_icl_prompt_fn()

        talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask = prepare_inputs(
            config=self.model_config,
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt,
            languages=[language],
            non_streaming_mode=False,
            text_embedding=self.text_embedding,
            input_embedding=self.input_embedding,
            text_projection=self.text_projection,
            device=self.device,
            voice_clone_spk_embeds=voice_clone_spk_embeds,
            generate_icl_prompt_fn=generate_icl_prompt_fn,
        )

        yield from self._generate_caller_driven(
            talker_input_embeds, trailing_text_hiddens, tts_pad_embed,
            str(uuid.uuid4()),
            SamplingParams(temperature=1.0, max_tokens=1),
            SamplingParams(temperature=temperature, top_p=top_p, max_tokens=17),
        )

    def _make_icl_prompt_fn(self):
        """Create ICL prompt function for voice cloning with reference audio."""
        text_embedding = self.text_embedding
        input_embedding = self.input_embedding
        text_projection = self.text_projection
        predictor_embeddings = self.predictor_input_embeddings
        config = self.model_config
        device = self.device

        def generate_icl_prompt(text_id, ref_id, ref_code, tts_pad_embed, tts_eos_embed, non_streaming_mode):
            text_embed = text_projection(text_embedding(torch.cat([ref_id, text_id], dim=-1)))
            text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)

            codec_embed_parts = [input_embedding(ref_code[:, :1])]
            for i in range(1, 16):
                codec_embed_parts.append(predictor_embeddings[i - 1](ref_code[:, i : i + 1]))
            codec_embed = torch.cat(codec_embed_parts, dim=1).sum(1).unsqueeze(0)

            codec_bos = input_embedding(
                torch.tensor([[config.talker_config.codec_bos_id]], device=device, dtype=text_id.dtype)
            )
            codec_embed = torch.cat([codec_bos, codec_embed], dim=1)

            text_lens, codec_lens = text_embed.shape[1], codec_embed.shape[1]

            if non_streaming_mode:
                pad_ids = torch.tensor(
                    [[config.talker_config.codec_pad_id] * text_lens], device=device, dtype=text_id.dtype
                )
                icl_embed = text_embed + input_embedding(pad_ids)
                return torch.cat([icl_embed, codec_embed + tts_pad_embed], dim=1), tts_pad_embed
            else:
                if text_lens > codec_lens:
                    return text_embed[:, :codec_lens] + codec_embed, text_embed[:, codec_lens:]
                else:
                    padding = [tts_pad_embed] * (codec_lens - text_lens)
                    text_embed_padded = torch.cat([text_embed] + padding, dim=1)
                    return text_embed_padded + codec_embed, tts_pad_embed

        return generate_icl_prompt

    async def generate_custom_voice_async(
        self, text: str, language: str = "English", speaker: str = "Vivian"
    ):
        """Async generator of codebook_id chunks. Requires zmq_bridge; call await start_zmq_tasks() first."""
        if self.zmq_bridge is None:
            raise RuntimeError("generate_custom_voice_async requires zmq_bridge")

        def _prep_in_thread() -> tuple:
            """Run prep in executor so event loop can run engine_loop; lock serializes GPU prep."""
            with self._prep_lock:
                input_ids, instruct_ids, speakers, languages = prepare_custom_voice_prompt(
                    text=text, language=language, speaker=speaker,
                    processor=self.processor, device=self.device,
                )
                return prepare_inputs(
                    config=self.model_config,
                    input_ids=input_ids, instruct_ids=instruct_ids, speakers=speakers, languages=languages,
                    non_streaming_mode=True,
                    text_embedding=self.text_embedding, input_embedding=self.input_embedding,
                    text_projection=self.text_projection, device=self.device,
                )

        loop = asyncio.get_event_loop()
        talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask = await loop.run_in_executor(
            None, _prep_in_thread
        )
        async for chunk in self.generate_async(
            talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask
        ):
            yield chunk

    def generate(self, inputs_embeds: torch.Tensor, trailing_text_hiddens: torch.Tensor, tts_pad_embed: torch.Tensor, talker_attention_mask: torch.Tensor, request_id: str | None = None):
        """Sync generator. Only valid when zmq_bridge is None."""
        if self.zmq_bridge is not None:
            raise RuntimeError("When using ZMQ bridge use generate_async() after await start_zmq_tasks()")
        request_id = request_id or str(uuid.uuid4())
        talker_sampling_params = SamplingParams(temperature=1.0, max_tokens=1)
        predictor_sampling_params = SamplingParams(temperature=0.9, max_tokens=17)
        yield from self._generate_caller_driven(
            inputs_embeds, trailing_text_hiddens, tts_pad_embed,
            request_id, talker_sampling_params, predictor_sampling_params,
        )

    async def generate_async(
        self,
        inputs_embeds: torch.Tensor,
        trailing_text_hiddens: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        talker_attention_mask: torch.Tensor,
        request_id: str | None = None,
    ):
        """Async generator of codebook_id chunks. ZMQ path; step() runs on event loop thread. Call await start_zmq_tasks() first."""
        if self.zmq_bridge is None:
            raise RuntimeError("generate_async requires zmq_bridge")
        talker_sampling_params = SamplingParams(temperature=1.0, max_tokens=1)
        predictor_sampling_params = SamplingParams(temperature=0.9, max_tokens=17)
        request_id = request_id or str(uuid.uuid4())
        request_queue: asyncio.Queue = asyncio.Queue()
        async with self._queues_lock:
            self._request_queues[request_id] = request_queue
        try:
            next_talker_embeds = inputs_embeds
            if next_talker_embeds.dim() == 2:
                next_talker_embeds = next_talker_embeds.unsqueeze(0)
            generation_step = 0
            self.talker_llm.add_request([next_talker_embeds], talker_sampling_params, request_id=request_id)

            while True:
                engine_type, msg_type, payload = await request_queue.get()
                if engine_type == "talker" and msg_type == "done":
                    self.talker_llm.clear_request(request_id)
                    break
                if engine_type == "talker" and msg_type == "token":
                    token_ids = payload["token_ids"]
                    hidden_states = payload.get("hidden_states")
                    last_id = token_ids[-1]
                    if last_id == 2150:
                        self.talker_llm.clear_request(request_id)
                        break
                    last_id_hidden = self.input_embedding(torch.tensor([last_id], device=self.device)).unsqueeze(0)
                    if hidden_states is not None:
                        h = torch.from_numpy(hidden_states.copy()).to(self.device)
                        if h.dim() == 1:
                            h = h.unsqueeze(0).unsqueeze(0)
                        else:
                            h = h.unsqueeze(0).unsqueeze(0)
                        last_hidden_state = h
                    else:
                        last_hidden_state = last_id_hidden.unsqueeze(0)
                    predictor_inputs_embeds = torch.cat((last_hidden_state, last_id_hidden), dim=1)
                    self.predictor_llm.add_request(
                        [predictor_inputs_embeds], predictor_sampling_params, request_id=request_id,
                    )
                    _, _, payload2 = await request_queue.get()
                    pred_token_ids = payload2.get("token_ids", [])
                    codebook_ids = [last_id] + pred_token_ids
                    yield codebook_ids

                    codec_hiddens = torch.cat(
                        [last_id_hidden]
                        + [self.predictor_input_embeddings[i](torch.tensor([pred_token_ids[i]], device=self.device)).unsqueeze(0) for i in range(15)],
                        dim=1,
                    )
                    next_talker_embeds = codec_hiddens.sum(1, keepdim=True)
                    if generation_step < trailing_text_hiddens.shape[1]:
                        next_talker_embeds = next_talker_embeds + trailing_text_hiddens[:, generation_step].unsqueeze(1)
                    else:
                        next_talker_embeds = next_talker_embeds + tts_pad_embed
                    generation_step += 1
                    self.talker_llm.add_request([next_talker_embeds], talker_sampling_params, request_id=request_id)
        finally:
            async with self._queues_lock:
                self._request_queues.pop(request_id, None)

    def _generate_caller_driven(
        self,
        inputs_embeds: torch.Tensor,
        trailing_text_hiddens: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        request_id: str,
        talker_sampling_params: SamplingParams,
        predictor_sampling_params: SamplingParams,
    ):
        generation_step = 0
        next_talker_embeds = inputs_embeds
        if next_talker_embeds.dim() == 2:
            next_talker_embeds = next_talker_embeds.unsqueeze(0)

        while True:
            self.talker_llm.add_request([next_talker_embeds], talker_sampling_params, request_id=request_id)
            _, _, outputs_all = self.talker_llm.step_with_outputs()
            if not outputs_all:
                self.talker_llm.clear_request(request_id)
                return

            match = next((o for o in outputs_all if o[0] == request_id), None)
            if match is None:
                continue
            _, _, token_ids, hidden_states, is_finished = match
            last_id = token_ids[-1]
            if last_id == 2150:
                self.talker_llm.clear_request(request_id)
                return

            last_id_hidden = self.input_embedding(torch.tensor([last_id], device=self.device)).unsqueeze(0)
            last_hidden_state = hidden_states.unsqueeze(0).unsqueeze(0)
            predictor_inputs_embeds = torch.cat((last_hidden_state, last_id_hidden), dim=1)
            predictor_outputs = self.predictor_llm.generate(
                [predictor_inputs_embeds.unsqueeze(0)],
                predictor_sampling_params,
                use_tqdm=False,
                request_id=request_id,
            )
            pred_token_ids = predictor_outputs[0]["token_ids"]
            codebook_ids = [last_id] + pred_token_ids
            yield codebook_ids

            codec_hiddens = torch.cat(
                [last_id_hidden]
                + [self.predictor_input_embeddings[i](torch.tensor([pred_token_ids[i]], device=self.device)).unsqueeze(0) for i in range(15)],
                dim=1,
            )
            next_talker_embeds = codec_hiddens.sum(1, keepdim=True)
            if generation_step < trailing_text_hiddens.shape[1]:
                next_talker_embeds = next_talker_embeds + trailing_text_hiddens[:, generation_step].unsqueeze(1)
            else:
                next_talker_embeds = next_talker_embeds + tts_pad_embed
            generation_step += 1


if __name__ == "__main__":
    interface = Qwen3TTSInterface(model_path="/work/weights/qwen3tts")
    print("Warm up...")
    audio_codes = list(interface.generate_custom_voice(text="Hi there this is a test.", language="English", speaker="Vivian"))

    print("Generate...")
    start = time.time()
    audio_codes = list(interface.generate_custom_voice(text="Hi there, this is tsdocode, hope you are doing well.", language="English", speaker="Vivian"))
    end = time.time()

    
    
    
    