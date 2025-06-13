"""Self-research training pipeline for the NeuroResearcher architecture.

This script orchestrates supervised pretraining and optional PPO fine-tuning
using the NeuroResearcher model. It includes meta-memory, prompt optimisation
and weighted language-model loss for reasoning tokens.
"""

from __future__ import annotations

import argparse
import os
import math
from typing import Any, Dict, List, Optional

try:
    import torch
    from torch import nn
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        GPT2Config,
        GPT2LMHeadModel,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        TrainerCallback,
        TrainerState,
        TrainerControl,
        AutoModelForSequenceClassification,
    )
    from trl import PPOTrainer, PPOConfig, create_reference_model
    from rich.console import Console
    from rich.progress import track
    from SelfResearch.data.dataset_loader import load_dataset_splits
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    import subprocess
    import sys

    print(
        f"Missing dependency: {exc}. Attempting install...\n"
        "Run `pip install transformers datasets trl rich matplotlib` if this fails."
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "transformers",
            "datasets",
            "trl",
            "rich",
            "matplotlib",
        ]
    )
    import torch
    from torch import nn
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        GPT2Config,
        GPT2LMHeadModel,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        TrainerCallback,
        TrainerState,
        TrainerControl,
        AutoModelForSequenceClassification,
    )
    from trl import PPOTrainer, PPOConfig, create_reference_model
    from rich.console import Console
    from rich.progress import track
    from SelfResearch.data.dataset_loader import load_dataset_splits
    import matplotlib.pyplot as plt

console = Console()


# ---------------------------------------------------------------------------
#                              META MEMORY
# ---------------------------------------------------------------------------
class MetaMemory:
    """Store hidden states across sessions."""

    def __init__(self) -> None:
        self._store: Dict[str, List[torch.Tensor]] = {}

    def add(self, session_id: str, state: torch.Tensor) -> None:
        self._store.setdefault(session_id, []).append(state.cpu())

    def get(self, session_id: str) -> List[torch.Tensor]:
        return self._store.get(session_id, [])


# ---------------------------------------------------------------------------
#                         NEURORESEARCHER MODEL
# ---------------------------------------------------------------------------
class NeuroResearcherConfig(GPT2Config):
    model_type = "neuroresearcher"

    def __init__(self, num_tools: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_tools = num_tools


class HFNeuroResearcherModel(GPT2LMHeadModel):
    config_class = NeuroResearcherConfig

    def __init__(
        self, config: NeuroResearcherConfig, *, enable_meta_memory: bool = False
    ) -> None:
        super().__init__(config)
        self.router = nn.Linear(config.n_embd, config.n_head * config.num_tools)
        self.meta_memory = MetaMemory() if enable_meta_memory else None
        self.enable_meta_memory = enable_meta_memory

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        session_id: Optional[str] = None,
        store_memory: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )
        last_hidden = outputs.hidden_states[-1]
        logits = outputs.logits
        router_logits = self.router(last_hidden[:, -1])
        router_weights = torch.softmax(
            router_logits.view(
                last_hidden.size(0), self.config.n_head, self.config.num_tools
            ),
            dim=-1,
        )
        if self.enable_meta_memory and store_memory and session_id:
            self.meta_memory.add(session_id, last_hidden.detach().mean(dim=1))
        return {
            "logits": logits,
            "loss": outputs.loss,
            "hidden_states": last_hidden,
            "tool_weights": router_weights,
        }


# ---------------------------------------------------------------------------
#                         SETUP FUNCTIONS
# ---------------------------------------------------------------------------


def setup_tokenizer(model_name: str) -> AutoTokenizer:
    """Load tokenizer and ensure padding token."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


def setup_model(
    tokenizer: AutoTokenizer,
    embed_dim: int,
    heads: int,
    layers: int,
    device: torch.device,
    gradient_ckpt: bool,
    enable_meta_memory: bool,
) -> HFNeuroResearcherModel:
    """Create and prepare the NeuroResearcher model."""
    config = NeuroResearcherConfig(
        vocab_size=len(tokenizer),
        n_embd=embed_dim,
        n_head=heads,
        n_layer=layers,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        n_positions=tokenizer.model_max_length,
    )
    model = HFNeuroResearcherModel(config, enable_meta_memory=enable_meta_memory)
    model.resize_token_embeddings(len(tokenizer))
    if gradient_ckpt:
        model.gradient_checkpointing_enable()
    model.to(device)
    return model


def prepare_datasets(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    train_split: str,
    eval_split: Optional[str],
    max_length: int,
) -> tuple[Any, Optional[Any]]:
    """Load tokenized train and optional evaluation datasets."""
    train_ds, eval_ds = load_dataset_splits(
        dataset_name,
        tokenizer.name_or_path,
        train_split=train_split,
        eval_split=eval_split,
        max_length=max_length,
    )
    return train_ds, eval_ds


# ---------------------------------------------------------------------------
#                         CUSTOM LOSS
# ---------------------------------------------------------------------------

_reason_s = None
_reason_e = None
_internal_s = None
_internal_e = None
_final_s = None
_plan_s = None
_plan_e = None
_analysis_s = None
_analysis_e = None

_ROUTER_WEIGHTS: List[torch.Tensor] = []
_HIDDEN_STATES: List[torch.Tensor] = []


def _init_tag_tokens(tokenizer: AutoTokenizer) -> None:
    """Initialize special token ID sequences for weight computation."""
    global _reason_s, _reason_e, _internal_s, _internal_e, _final_s, _plan_s, _plan_e, _analysis_s, _analysis_e
    _reason_s = tokenizer.encode("<reasoning>", add_special_tokens=False)
    _reason_e = tokenizer.encode("</reasoning>", add_special_tokens=False)
    _internal_s = tokenizer.encode("<internal_thinking>", add_special_tokens=False)
    _internal_e = tokenizer.encode("</internal_thinking>", add_special_tokens=False)
    _final_s = tokenizer.encode("<final_output>", add_special_tokens=False)
    _plan_s = tokenizer.encode("<plan>", add_special_tokens=False)
    _plan_e = tokenizer.encode("</plan>", add_special_tokens=False)
    _analysis_s = tokenizer.encode("<analysis>", add_special_tokens=False)
    _analysis_e = tokenizer.encode("</analysis>", add_special_tokens=False)


def _apply_weights(ids: List[int]) -> List[float]:
    """Return token weights for a sequence of IDs."""
    weights = [1.0] * len(ids)
    i = 0
    while i < len(ids):
        if ids[i : i + len(_reason_s)] == _reason_s:
            j = i + len(_reason_s)
            while j < len(ids) and ids[j : j + len(_reason_e)] != _reason_e:
                weights[j] = 1.5
                j += 1
            i = j
        elif ids[i : i + len(_plan_s)] == _plan_s:
            j = i + len(_plan_s)
            while j < len(ids) and ids[j : j + len(_plan_e)] != _plan_e:
                weights[j] = 1.3
                j += 1
            i = j
        elif ids[i : i + len(_analysis_s)] == _analysis_s:
            j = i + len(_analysis_s)
            while j < len(ids) and ids[j : j + len(_analysis_e)] != _analysis_e:
                weights[j] = 1.4
                j += 1
            i = j
        elif ids[i : i + len(_internal_s)] == _internal_s:
            j = i + len(_internal_s)
            while j < len(ids) and ids[j : j + len(_internal_e)] != _internal_e:
                weights[j] = 1.2
                j += 1
            i = j
        elif ids[i : i + len(_final_s)] == _final_s:
            for j in range(i, len(ids)):
                weights[j] = 1.0
            break
        else:
            i += 1
    return weights


def compute_loss(
    model: HFNeuroResearcherModel, inputs: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Compute weighted LM loss.

    Tokens inside ``<reasoning>`` blocks receive weight ``1.5``,
    ``<plan>`` blocks weight ``1.3``, ``<analysis>`` blocks weight ``1.4``,
    ``<internal_thinking>`` blocks weight ``1.2``, and ``<final_output>``
    resets to ``1.0`` for all remaining tokens.
    """
    input_ids = inputs["input_ids"] if "input_ids" in inputs else inputs["labels"]
    attention_mask = inputs.get("attention_mask")
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
        session_id=_SESSION_ID,
    )
    _ROUTER_WEIGHTS.append(outputs["tool_weights"].detach().cpu())
    _HIDDEN_STATES.append(outputs["hidden_states"].detach().mean(dim=1).cpu())
    logits = outputs["logits"]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    losses = losses.view(shift_labels.size())
    weights = []
    for row in shift_labels:
        ids = row.tolist()
        weights.append(_apply_weights(ids))
    weight_tensor = torch.tensor(weights, device=losses.device)
    mask = shift_labels.ne(-100)
    weighted = losses * weight_tensor * mask
    return weighted.sum() / mask.sum()


# ---------------------------------------------------------------------------
#                        CALLBACK FOR PROMPT UPDATES
# ---------------------------------------------------------------------------
class CosinePromptUpdateCallback(TrainerCallback):
    """Periodically optimise the base prompt using a cosine schedule."""

    def __init__(self, optimizer: Any, total_epochs: int, base_prompt: str) -> None:
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.current_prompt = base_prompt

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if state.epoch is None:
            return
        epoch = int(state.epoch) + 1
        interval = max(1, int(1 + math.cos(math.pi * epoch / self.total_epochs) * 2))
        if epoch % interval == 0:
            self.current_prompt = self.optimizer.optimize_prompt(self.current_prompt)
            console.print(
                f"[PromptUpdateCallback] Updated prompt: {self.current_prompt}"
            )


class EarlyStopper(TrainerCallback):
    """Stop training if evaluation loss does not improve."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_loss: Optional[float] = None
        self.counter = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs: Any,
    ) -> None:
        if self.patience <= 0 or "eval_loss" not in metrics:
            return
        loss = metrics["eval_loss"]
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                control.should_training_stop = True


# ---------------------------------------------------------------------------
#                        TRAINING PHASES
# ---------------------------------------------------------------------------


def run_supervised(
    model: HFNeuroResearcherModel,
    tokenizer: AutoTokenizer,
    train_ds: Any,
    eval_ds: Optional[Any],
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    grad_accum: int,
    warmup_steps: int,
    fp16: bool,
    bf16: bool,
    device: torch.device,
    *,
    patience: int = 0,
    debug: bool = False,
) -> List[float]:
    """Run supervised LM training and return loss history.

    If ``eval_ds`` is provided, evaluation occurs at the end of each epoch.
    """
    _init_tag_tokens(tokenizer)
    optimizer = MetaPromptOptimizer(tokenizer.name_or_path, device)
    base_prompt = f"Session { _SESSION_ID }"
    if len(train_ds) > 0 and "text" in train_ds[0]:
        sample_text = str(train_ds[0]["text"])[:50]
        base_prompt += f" | {sample_text}"
    prompt_cb = CosinePromptUpdateCallback(optimizer, epochs, base_prompt)
    if debug:
        train_ds = train_ds.select(range(min(100, len(train_ds))))
        if eval_ds is not None:
            eval_ds = eval_ds.select(range(min(100, len(eval_ds))))
    _ROUTER_WEIGHTS.clear()
    _HIDDEN_STATES.clear()
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=warmup_steps,
        fp16=fp16,
        bf16=bf16,
        logging_steps=1 if debug else 10,
        save_steps=500,
        report_to="none",
        evaluation_strategy="epoch" if eval_ds is not None else "no",
    )
    lm_losses: List[float] = []

    class LossTracker(TrainerCallback):
        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: Dict[str, float],
            **kwargs: Any,
        ) -> None:
            if "loss" in logs:
                lm_losses.append(logs["loss"])

    callbacks: List[TrainerCallback] = [prompt_cb, LossTracker()]
    if patience > 0:
        callbacks.append(EarlyStopper(patience))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_loss=compute_loss,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return lm_losses


def run_ppo(
    model: HFNeuroResearcherModel,
    tokenizer: AutoTokenizer,
    dataset: Any,
    output_dir: str,
    epochs: int,
    batch_size: int,
    clip: float,
    kl_coef: float,
    device: torch.device,
    *,
    debug: bool = False,
) -> List[float]:
    """Fine-tune the model with PPO using a frozen reward model."""
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "ayjays132/NR1-rm"
    ).to(device)
    for param in reward_model.parameters():
        param.requires_grad = False
    reward_model.eval()

    ppo_config = PPOConfig(
        batch_size=batch_size,
        learning_rate=5e-5,
        cliprange=clip,
        kl_coef=kl_coef,
    )
    ref_model = create_reference_model(model)
    ppo_trainer = PPOTrainer(
        ppo_config,
        policy_model=model,
        value_model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
    )
    rewards: List[float] = []
    step = 0

    if debug:
        dataset = dataset.select(range(min(32, len(dataset))))
    for epoch in range(epochs):
        epoch_data = dataset.shuffle(seed=epoch) if hasattr(dataset, "shuffle") else dataset
        for sample in epoch_data:
            prompt = (
                sample.get("prompt")
                or sample.get("text")
                or sample.get("question")
                or ""
            )[:256]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    top_p=0.95,
                )
            response = tokenizer.decode(
                generated[0][inputs["input_ids"].size(1) :], skip_special_tokens=True
            )
            rm_inp = tokenizer(prompt + response, return_tensors="pt").to(device)
            with torch.no_grad():
                rm_out = reward_model(**rm_inp)
                reward = float(rm_out.logits.squeeze()[0])
            ppo_trainer.step([prompt], [response], rewards=[reward])
            rewards.append(reward)
            step += 1
            if step % 10 == 0:
                console.print(f"[PPO] step {step}: reward={reward:.4f}")

    ppo_trainer.save_pretrained(output_dir)
    return rewards


# ---------------------------------------------------------------------------
#                           SELF-PLAY GENERATION
# ---------------------------------------------------------------------------

def run_self_play(
    model: HFNeuroResearcherModel,
    tokenizer: AutoTokenizer,
    dataset: Any,
    output_dir: str,
    episodes: int,
    turns: int,
    device: torch.device,
    *,
    debug: bool = False,
    optimize_prompts: bool = False,
) -> List[str]:
    """Generate conversation transcripts via self-play.

    When ``optimize_prompts`` is ``True``, the :class:`MetaPromptOptimizer`
    refines the initial prompt of each episode.
    """
    transcripts: List[str] = []
    optimizer = None
    if optimize_prompts:
        optimizer = MetaPromptOptimizer(tokenizer.name_or_path, device)
    if debug:
        episodes = min(episodes, 1)
        dataset = dataset.select(range(min(10, len(dataset))))
    for i in track(range(episodes), description="Self-play"):
        sample = dataset.shuffle(seed=i)[0] if hasattr(dataset, "shuffle") else dataset[i % len(dataset)]
        prompt = (
            sample.get("prompt")
            or sample.get("text")
            or sample.get("question")
            or "Discuss research."
        )
        if optimizer is not None:
            prompt = optimizer.optimize_prompt(prompt)
        dialogue = prompt
        for _ in range(turns):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    top_p=0.95,
                )
            prompt = tokenizer.decode(generated[0], skip_special_tokens=True)
            dialogue += "\n" + prompt
        transcripts.append(dialogue)
    path = os.path.join(output_dir, "self_play_transcripts.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n\n".join(transcripts))
    console.print(f"Self-play transcripts saved to {path}")
    return transcripts


# ---------------------------------------------------------------------------
#                         METRICS & PLOTTING
# ---------------------------------------------------------------------------


def plot_metrics(
    losses: List[float],
    rewards: List[float],
    weights: List[torch.Tensor],
    hidden_states: List[torch.Tensor],
    output_dir: str,
) -> None:
    """Save training plots to disk."""
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    steps = list(range(1, len(losses) + 1))
    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    loss_path = os.path.join(output_dir, "plots", "lm_loss.png")
    plt.savefig(loss_path)
    plt.close()

    if rewards:
        r_steps = list(range(1, len(rewards) + 1))
        plt.figure()
        plt.plot(r_steps, rewards)
        plt.xlabel("PPO step")
        plt.ylabel("Reward")
        rew_path = os.path.join(output_dir, "plots", "ppo_rewards.png")
        plt.savefig(rew_path)
        plt.close()
    else:
        rew_path = ""

    if weights:
        avg_weights = torch.stack(weights).mean(dim=0).mean(dim=0).cpu().numpy()
        plt.figure()
        for h, head_w in enumerate(avg_weights):
            plt.plot(head_w, label=f"head {h}")
        plt.xlabel("Tool index")
        plt.ylabel("Avg weight")
        plt.legend()
        w_path = os.path.join(output_dir, "plots", "tool_weights.png")
        plt.savefig(w_path)
        plt.close()
    else:
        w_path = ""

    if hidden_states:
        norms = [hs.norm(dim=1).mean().item() for hs in hidden_states]
        n_steps = list(range(1, len(norms) + 1))
        plt.figure()
        plt.plot(n_steps, norms)
        plt.xlabel("Training step")
        plt.ylabel("Hidden state norm")
        h_path = os.path.join(output_dir, "plots", "hidden_state_norms.png")
        plt.savefig(h_path)
        plt.close()
    else:
        h_path = ""

    console.print(f"Loss plot saved to {loss_path}")
    if rew_path:
        console.print(f"Reward plot saved to {rew_path}")
    if w_path:
        console.print(f"Tool weight plot saved to {w_path}")
    if h_path:
        console.print(f"Hidden state plot saved to {h_path}")


# ---------------------------------------------------------------------------
#                            ENVIRONMENT SETUP
# ---------------------------------------------------------------------------

def setup_environment() -> torch.device:
    """Return PyTorch device and enable GPU features when available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
    return device


# ---------------------------------------------------------------------------
#                                 CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Train the NeuroResearcher model")
    parser.add_argument("--dataset", type=str, default="ag_news", help="Dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--eval-split", type=str, default=None, help="Evaluation split")
    parser.add_argument(
        "--model-path",
        type=str,
        default="ayjays132/NeuroReasoner-1-NR-1",
        help="Pretrained model path",
    )
    parser.add_argument("--output-dir", type=str, default="./nr_outputs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--ppo-epochs", type=int, default=0)
    parser.add_argument("--ppo-batch-size", type=int, default=1)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--ppo-kl", type=float, default=0.1)
    parser.add_argument("--self-play-episodes", type=int, default=0)
    parser.add_argument("--self-play-turns", type=int, default=4)
    parser.add_argument(
        "--optimize-self-play",
        action="store_true",
        help="Use MetaPromptOptimizer during self-play",
    )
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--session-id", type=str, required=True, help="Unique identifier for meta-memory")
    parser.add_argument(
        "--no-meta-memory",
        action="store_true",
        help="Disable meta-memory storage during training",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience (0 to disable)",
    )
    return parser.parse_args()


# Global session id for compute_loss
_SESSION_ID: str = ""


# ---------------------------------------------------------------------------
#                                MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    global _SESSION_ID
    _SESSION_ID = args.session_id
    device = setup_environment()
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        if os.path.isdir(args.model_path) and not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model path {args.model_path} not found")
        tokenizer = setup_tokenizer(args.model_path)
        model = setup_model(
            tokenizer,
            args.embed_dim,
            args.heads,
            args.layers,
            device,
            args.gradient_checkpointing,
            enable_meta_memory=not args.no_meta_memory,
        )
        train_ds, eval_ds = prepare_datasets(
            args.dataset,
            tokenizer,
            args.split,
            args.eval_split,
            tokenizer.model_max_length,
        )
        console.print(f"Loaded {len(train_ds)} training examples")
        if eval_ds is not None:
            console.print(f"Loaded {len(eval_ds)} evaluation examples")
    except Exception as exc:
        console.print(f"[red]Initialization failed: {exc}")
        return

    console.print("[bold green]Starting supervised training...")
    losses = run_supervised(
        model,
        tokenizer,
        train_ds,
        eval_ds,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.lr,
        args.grad_accum,
        args.warmup_steps,
        args.fp16,
        args.bf16,
        device,
        patience=args.patience,
        debug=args.debug,
    )

    rewards: List[float] = []
    if args.ppo_epochs > 0:
        console.print("[bold green]Starting PPO fine-tuning...")
        rewards = run_ppo(
            model,
            tokenizer,
            train_ds,
            args.output_dir,
            args.ppo_epochs,
            args.ppo_batch_size,
            args.ppo_clip,
            args.ppo_kl,
            device,
            debug=args.debug,
        )

    if args.self_play_episodes > 0:
        console.print("[bold green]Generating self-play transcripts...")
        run_self_play(
            model,
            tokenizer,
            train_ds,
            args.output_dir,
            args.self_play_episodes,
            args.self_play_turns,
            device,
            debug=args.debug,
            optimize_prompts=args.optimize_self_play,
        )

    plot_metrics(losses, rewards, _ROUTER_WEIGHTS, _HIDDEN_STATES, args.output_dir)
    console.print(
        f"\n[green]🎉 Training complete — all artifacts saved under {args.output_dir}"
    )


if __name__ == "__main__":
    main()
