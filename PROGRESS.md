# PA2 Progress — COMPLETE

## All Tasks Done
- [x] Task 1: Repo setup, smoke tests pass (7/7)
- [x] Task 2: Baseline generation (AlpacaEval + HH-RLHF)
- [x] Task 3: SFT training (30k examples, 1 epoch, val_loss=341.34)
- [x] Task 4a: DPO HH-RLHF (reward_acc=0.617, margin=0.187)
- [x] Task 4b: DPO UltraFeedback (reward_acc=0.812, margin=1.84)
- [x] Task 5: LoRA DPO Persona (Ramsay, 3 epochs, 2 min)
- [x] Task 6: Generate eval outputs (SFT vs DPO pairs for both tracks)
- [x] Task 7: Model-based judging (300/300 both tracks)
- [x] Report: Complete with all results

## Final Judging Results

### UltraFeedback (300/300)
| | Count |
|---|---|
| DPO wins | 120 |
| SFT wins | 121 |
| Ties | 59 |
| DPO win rate | 49.8% |

### HH-RLHF (300/300)
| | Count |
|---|---|
| DPO wins | 113 |
| SFT wins | 86 |
| Ties | 99 |
| Unknown | 2 |
| DPO win rate | 56.8% |

## Key Files
- Report: `report/report.tex`
- Judgments: `outputs/judgments/`
- Eval pairs: `outputs/eval_pairs/`

## Git Remote
Push with: `git push mine main`
