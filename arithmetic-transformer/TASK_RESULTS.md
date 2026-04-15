## Literature Summary

- DeepSeekMath motivates the order used here: first build base competence, then supervised reasoning traces, then reinforcement learning with verifiable rewards.
- Process supervision matters because final-answer-only reward can still permit shortcutting or malformed intermediate traces.
- Group-relative policy updates are a reasonable fit for small symbolic tasks because they compare several sampled trajectories for the same prompt.
- KL-style anchoring or a reference-policy penalty helps protect base competence during RL-style updates.
- Short validation runs before scaling up are important; this framework keeps each phase compact and inspectable before any larger run.

Sources:
- DeepSeekMath: https://arxiv.org/abs/2402.03300
- Hugging Face TRL GRPO docs: https://huggingface.co/docs/trl/main/en/grpo_trainer


# Phase Results

## phase0_baseline / answer_sft_only
- decision: `unclear`
- checkpoint: `runs\rlvr_tiny\phase0_baseline\answer_sft_only\phase_checkpoint.pt`
- pre final-answer acc: `0.0000`
- post SFT final-answer acc: `0.0000`
- final-answer acc after final step: `0.0000`
- parseable trace rate after final step: `1.0000`
- full-step-valid rate after final step: `0.0000`
- average valid-step fraction after final step: `0.0000`
- average output length after final step: `12.31`

## phase1_final_1d1d / trace_sft_only
- decision: `unclear`
- checkpoint: `runs\rlvr_tiny\phase1_final_1d1d\trace_sft_only\phase_checkpoint.pt`
- pre final-answer acc: `0.0000`
- post SFT final-answer acc: `0.0000`
- final-answer acc after final step: `0.0000`
- parseable trace rate after final step: `0.0938`
- full-step-valid rate after final step: `0.0000`
- average valid-step fraction after final step: `0.0000`
- average output length after final step: `4.09`

## phase1_final_1d1d / trace_sft_rlvr
- decision: `unclear`
- checkpoint: `runs\rlvr_tiny\phase1_final_1d1d\trace_sft_rlvr\phase_checkpoint.pt`
- pre final-answer acc: `0.0000`
- post SFT final-answer acc: `0.0625`
- final-answer acc after final step: `0.0625`
- parseable trace rate after final step: `1.0000`
- full-step-valid rate after final step: `0.0625`
- average valid-step fraction after final step: `0.0625`
- average output length after final step: `5.00`

## phase2_trace_2d1d / answer_sft_only
- decision: `unclear`
- checkpoint: `runs\rlvr_tiny\phase2_trace_2d1d\answer_sft_only\phase_checkpoint.pt`
- pre final-answer acc: `0.0000`
- post SFT final-answer acc: `0.0000`
- final-answer acc after final step: `0.0000`
- parseable trace rate after final step: `0.0000`
- full-step-valid rate after final step: `0.0000`
- average valid-step fraction after final step: `0.0000`
- average output length after final step: `5.00`

## phase2_trace_2d1d / trace_sft_only
- decision: `unclear`
- checkpoint: `runs\rlvr_tiny\phase2_trace_2d1d\trace_sft_only\phase_checkpoint.pt`
- pre final-answer acc: `0.0000`
- post SFT final-answer acc: `0.0000`
- final-answer acc after final step: `0.0000`
- parseable trace rate after final step: `1.0000`
- full-step-valid rate after final step: `0.0000`
- average valid-step fraction after final step: `0.0000`
- average output length after final step: `24.00`

## phase2_trace_2d1d / trace_sft_rlvr
- decision: `unclear`
- checkpoint: `runs\rlvr_tiny\phase2_trace_2d1d\trace_sft_rlvr\phase_checkpoint.pt`
- pre final-answer acc: `0.0000`
- post SFT final-answer acc: `0.0000`
- final-answer acc after final step: `0.0000`
- parseable trace rate after final step: `1.0000`
- full-step-valid rate after final step: `0.0000`
- average valid-step fraction after final step: `0.0000`
- average output length after final step: `6.00`
- RLVR steps: `10`

## phase2_trace_2d1d / trace_sft_best_of_n
- decision: `unclear`
- checkpoint: `runs\rlvr_tiny\phase2_trace_2d1d\trace_sft_best_of_n\phase_checkpoint.pt`
- pre final-answer acc: `0.0000`
- post SFT final-answer acc: `0.0000`
- final-answer acc after final step: `0.0000`
- parseable trace rate after final step: `1.0000`
- full-step-valid rate after final step: `0.0000`
- average valid-step fraction after final step: `0.0000`
- average output length after final step: `24.00`
- best-of-N steps: `6`

## Recommendation
Keep phase ordering curriculum-first, prefer trace SFT before RLVR, and compare against best-of-N on nontrivial phases because small models may benefit almost as much from self-training as from policy optimization.
