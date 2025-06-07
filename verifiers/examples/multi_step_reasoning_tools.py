import verifiers as vf
from verifiers.envs.reasoninggym_env import ReasoningGymEnv
from verifiers.envs.tool_env import ToolEnv
from verifiers.prompts.system_prompts import DEFAULT_TOOL_PROMPT_TEMPLATE
from verifiers.tools import python, calculator

"""
Multi-GPU training (single node, 4 training + 4 inference) using ToolEnv
with the multi_step_reasoning task from reasoning-gym.

CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_server.py \
    --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml \
    --num-processes 4 verifiers/examples/multi_step_reasoning_tools.py
"""

# Build the ReasoningGym dataset for multi_step_reasoning
rg_env = ReasoningGymEnv(
    gym="multi_step_reasoning",
    num_samples=2000,
    num_eval_samples=200,
    max_concurrent=128,
)

vf_env = ToolEnv(
    dataset=rg_env.dataset,
    eval_dataset=rg_env.eval_dataset,
    system_prompt=DEFAULT_TOOL_PROMPT_TEMPLATE,
    few_shot=[],
    tools=[python, calculator],
    max_turns=5,
)
print(vf_env.system_prompt)

model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "multi_step_reasoning-grpo_" + model_name.split("/")[-1].lower()

args = vf.grpo_defaults(run_name=run_name)
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args,
)
trainer.train()
