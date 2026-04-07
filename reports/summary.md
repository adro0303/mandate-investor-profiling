# Summary

- Dataset: `data\processed\ml_dataset.parquet`
- Mode: `quick`
- Walk-forward: start=0.7, step=20, test_window=60
- Best model (directional_accuracy): `mlp`
- Best global by RMSE: `baseline_zero`
- Best global by directional_accuracy: `mlp`
- Targets where baseline is best by RMSE: 15; trained model best: 0
- Targets where baseline is best by directional_accuracy: 0; trained model best: 15

## Leakage checks

- In walk-forward evaluation, each fold retrains from scratch per target.
- Imputation/scaling is fit on the fold train split only (never on test).
- Trained artifacts are not reused across folds inside evaluate.
- run_all without `--force` skips recomputation using existing CSVs/models; it does not mix data across time.

## Global model table

| model | avg_mae | avg_rmse | avg_r2 | avg_directional_accuracy | avg_hit_rate | avg_pearson | avg_spearman | total_valid_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | 0.0228883344703869 | 0.0319625490335015 | -13.452665375276624 | 0.4899658074264565 | 0.4899658074264565 | -0.0027287502557656 | -0.0061548304592528 | 13022 |
| rf | 0.0088551238729393 | 0.0116861026670373 | -0.1019484616654653 | 0.4875965795445173 | 0.4875965795445173 | -0.0082572549901982 | -0.0142966579465776 | 13022 |
| knn | 0.0090293978607056 | 0.0119434919970747 | -0.15206849971916 | 0.4855679266984202 | 0.4855679266984202 | -0.0057392126677784 | 0.0081709792737298 | 13022 |
| ridge | 0.0248648945621813 | 0.0359232098265904 | -14.2120861564722 | 0.4668571287377575 | 0.4668571287377575 | -0.0517784618521798 | -0.0750959857594981 | 13022 |
| baseline_last | 0.0114785272928434 | 0.0151995323133119 | -0.8004446507691156 | 0.3444446456365658 | 0.3444446456365658 | -0.034014572336265 | -0.0588288588576851 | 13022 |
| baseline_zero | 0.0085156153189018 | 0.0113588785980272 | -0.0277360556846182 | 0.0079654170731587 | 0.0079654170731587 | nan | nan | 13022 |

## Practical significance of directional accuracy

| model | directional_accuracy_mean | directional_accuracy_std_target | directional_accuracy_std_fold | excess_over_50 |
| --- | --- | --- | --- | --- |
| mlp | 0.48996580742645646 | 0.01379963245083666 | 0.021711399406092992 | -0.010034192573543543 |
| rf | 0.4875965795445173 | 0.04212071672506136 | 0.03072740188151983 | -0.012403420455482705 |
| knn | 0.4855679266984202 | 0.020654379325366002 | 0.038184506581174026 | -0.014432073301579806 |
| ridge | 0.4668571287377574 | 0.017195741161630095 | 0.028750325798750375 | -0.03314287126224258 |
| baseline_last | 0.34444464563656585 | 0.048897255253687 | 0.03139402244886839 | -0.15555535436343415 |
| baseline_zero | 0.007965417073158775 | 0.00819818053207588 | 0.004756264356678751 | -0.4920345829268412 |

## Target distribution diagnostics

| target | n_valid | mean | std | pos_ratio | neg_ratio | zero_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| y_BTC-USD | 1805 | 0.0009346113657441722 | 0.030186598773608398 | 0.4969529085872576 | 0.5030470914127424 | 0.0 |
| y_GLD | 991 | 0.0008257250420941282 | 0.010576705902315407 | 0.5428859737638748 | 0.45610494450050454 | 0.0010090817356205853 |
| y_TLT | 991 | 0.0002839809643422279 | 0.009957527076167599 | 0.5146316851664985 | 0.48234106962663975 | 0.0030272452068617556 |
| y_UUP | 991 | 0.00015230251005816966 | 0.004580602274537166 | 0.5176589303733602 | 0.44601412714429867 | 0.03632694248234107 |
| y_XLB | 991 | 0.0005320986000283205 | 0.012248425200807782 | 0.520686175580222 | 0.47426841574167505 | 0.005045408678102927 |
| y_XLC | 991 | 0.0004888327949035201 | 0.01352651397112826 | 0.5277497477295661 | 0.4651866801210898 | 0.007063572149344097 |
| y_XLE | 991 | 0.0010595190458911857 | 0.016349059644948614 | 0.5519677093844602 | 0.4429868819374369 | 0.005045408678102927 |
| y_XLF | 991 | 0.0006672946388741781 | 0.012081862622604328 | 0.5267406659939455 | 0.46215943491422806 | 0.011099899091826439 |
| y_XLI | 991 | 0.0006419037509563635 | 0.011164323431795395 | 0.5307769929364279 | 0.4661957618567104 | 0.0030272452068617556 |
| y_XLK | 991 | 0.0007004596074113018 | 0.015989162605895786 | 0.5408678102926338 | 0.4581231079717457 | 0.0010090817356205853 |
| y_XLP | 991 | 0.0003292426142711241 | 0.008438513746745415 | 0.5257315842583249 | 0.4661957618567104 | 0.008072653884964682 |
| y_XLRE | 991 | 0.0005204516372394256 | 0.012196045876949845 | 0.5196770938446014 | 0.46417759838546924 | 0.016145307769929364 |
| y_XLU | 991 | 0.0006885138485530835 | 0.010829128365983697 | 0.5277497477295661 | 0.4611503531786075 | 0.011099899091826439 |
| y_XLV | 991 | 0.0003698070752108559 | 0.009298710748560765 | 0.5216952573158425 | 0.47628657921291623 | 0.0020181634712411706 |
| y_XLY | 991 | 0.0002732078391204174 | 0.015282656431501734 | 0.5216952573158425 | 0.47628657921291623 | 0.0020181634712411706 |

## Per-target table (model vs metrics)

| target | model | rmse_mean | directional_accuracy_mean | n_samples |
| --- | --- | --- | --- | --- |
| y_BTC-USD | baseline_zero | 0.0233706510692567 | 0.0 | 1500 |
| y_BTC-USD | rf | 0.0240361779433488 | 0.4826666666666666 | 1500 |
| y_BTC-USD | knn | 0.0252203479153699 | 0.4946666666666666 | 1500 |
| y_BTC-USD | baseline_last | 0.03349788064413 | 0.4933333333333333 | 1500 |
| y_BTC-USD | mlp | 0.0360172609042712 | 0.4753333333333333 | 1500 |
| y_BTC-USD | ridge | 0.0471887112313881 | 0.4673333333333333 | 1500 |
| y_GLD | baseline_zero | 0.01168634575814 | 0.0 | 823 |
| y_GLD | rf | 0.0118340634591337 | 0.5094850551731891 | 823 |
| y_GLD | knn | 0.0119825143279547 | 0.4363616795913956 | 823 |
| y_GLD | baseline_last | 0.0152408227919874 | 0.3187361858447051 | 823 |
| y_GLD | ridge | 0.0256333909275882 | 0.4495346548176162 | 823 |
| y_GLD | mlp | 0.0316974138372037 | 0.5036658754392019 | 823 |
| y_TLT | baseline_zero | 0.0073000618233073 | 0.0035314489432136 | 823 |
| y_TLT | rf | 0.0073534076106645 | 0.5424672029053368 | 823 |
| y_TLT | knn | 0.0079374087933537 | 0.4832852846656091 | 823 |
| y_TLT | baseline_last | 0.0096391102850792 | 0.3473534052017824 | 823 |
| y_TLT | mlp | 0.0317348732351321 | 0.5159486941961586 | 823 |
| y_TLT | ridge | 0.0327168564415851 | 0.4700585826402256 | 823 |
| y_UUP | baseline_zero | 0.0044089311675435 | 0.0286517418079284 | 823 |
| y_UUP | rf | 0.0044740959689597 | 0.5139594506191869 | 823 |
| y_UUP | knn | 0.004601648937429 | 0.50561208606594 | 823 |
| y_UUP | baseline_last | 0.0058875336499423 | 0.3163699308750018 | 823 |
| y_UUP | ridge | 0.0137925460001989 | 0.4499136453832195 | 823 |
| y_UUP | mlp | 0.0285083561754081 | 0.4950323388376126 | 823 |
| y_XLB | baseline_zero | 0.0112356316226326 | 0.0119449961802903 | 823 |
| y_XLB | rf | 0.0118517831918983 | 0.3986669503728327 | 823 |
| y_XLB | knn | 0.0118736643906375 | 0.490097870340771 | 823 |
| y_XLB | baseline_last | 0.0151815794194969 | 0.3458560526557484 | 823 |
| y_XLB | mlp | 0.0295012882463318 | 0.4795357362051073 | 823 |
| y_XLB | ridge | 0.0388488100259048 | 0.451701231169791 | 823 |
| y_XLC | baseline_zero | 0.009985373387897 | 0.0067335907335907 | 823 |
| y_XLC | rf | 0.0102856877677283 | 0.5531779060612732 | 823 |
| y_XLC | knn | 0.010687931005215 | 0.4754318295667181 | 823 |
| y_XLC | baseline_last | 0.0128300969132568 | 0.3893064368008587 | 823 |
| y_XLC | mlp | 0.0306461591118201 | 0.48649947262929 | 823 |
| y_XLC | ridge | 0.0416915922252662 | 0.4808690301560483 | 823 |
| y_XLE | baseline_zero | 0.0142360083153374 | 0.011372264126828 | 823 |
| y_XLE | rf | 0.0147487655867738 | 0.5069244536283076 | 823 |
| y_XLE | knn | 0.0147915150515334 | 0.4764126842985869 | 823 |
| y_XLE | baseline_last | 0.0186272145487794 | 0.3608439582465952 | 823 |
| y_XLE | mlp | 0.0321924887178446 | 0.504334065895932 | 823 |
| y_XLE | ridge | 0.0410893134166333 | 0.5109376985781651 | 823 |
| y_XLF | baseline_zero | 0.0111594716929655 | 0.010453781512605 | 823 |
| y_XLF | rf | 0.0114935552814725 | 0.469504277511884 | 823 |
| y_XLF | knn | 0.011754602231038 | 0.4816213436243862 | 823 |
| y_XLF | baseline_last | 0.0142254569312567 | 0.3241467854409031 | 823 |
| y_XLF | mlp | 0.0305531741631546 | 0.4815347338460928 | 823 |
| y_XLF | ridge | 0.0353964751095192 | 0.4572674691036759 | 823 |
| y_XLI | baseline_zero | 0.0105122593240069 | 0.0 | 823 |
| y_XLI | rf | 0.0108421389079392 | 0.4473913378948876 | 823 |
| y_XLI | knn | 0.0112113840695856 | 0.5077713642490519 | 823 |
| y_XLI | baseline_last | 0.0137826148748528 | 0.3046477700007111 | 823 |
| y_XLI | mlp | 0.0317638834840453 | 0.4807445560270104 | 823 |
| y_XLI | ridge | 0.0324230395956207 | 0.4510578412571313 | 823 |
| y_XLK | baseline_zero | 0.0151066318702666 | 0.0 | 823 |
| y_XLK | rf | 0.0154744289467148 | 0.4680253546992897 | 823 |
| y_XLK | knn | 0.0157247812200919 | 0.4802654805367787 | 823 |
| y_XLK | baseline_last | 0.0201914787041802 | 0.3493916258333094 | 823 |
| y_XLK | mlp | 0.0337117030634266 | 0.464754308963741 | 823 |
| y_XLK | ridge | 0.0485376859102177 | 0.495132721345196 | 823 |
| y_XLP | baseline_zero | 0.0080375106277312 | 0.0143265983509391 | 823 |
| y_XLP | knn | 0.0081325660407685 | 0.4753992635980466 | 823 |
| y_XLP | rf | 0.0082439815105862 | 0.4792203294150555 | 823 |
| y_XLP | baseline_last | 0.0106641720473542 | 0.2976059781364041 | 823 |
| y_XLP | ridge | 0.0277037379431664 | 0.4641931860009953 | 823 |
| y_XLP | mlp | 0.0300887483292708 | 0.4987768662342698 | 823 |
| y_XLRE | baseline_zero | 0.0099546676191726 | 0.020133933570039 | 823 |
| y_XLRE | rf | 0.0102895494756859 | 0.5293094051745166 | 823 |
| y_XLRE | knn | 0.0106377790404456 | 0.4986380921700394 | 823 |
| y_XLRE | baseline_last | 0.0135873862255798 | 0.3057789906816276 | 823 |
| y_XLRE | mlp | 0.032159200012729 | 0.5056437130144027 | 823 |
| y_XLRE | ridge | 0.0406004331281908 | 0.4534006794721804 | 823 |
| y_XLU | baseline_zero | 0.0099258479017048 | 0.0087321878594698 | 823 |
| y_XLU | rf | 0.0101755284158684 | 0.4988362714417481 | 823 |
| y_XLU | knn | 0.0102794022166652 | 0.525855758856773 | 823 |
| y_XLU | baseline_last | 0.0129041708411758 | 0.3444115695540645 | 823 |
| y_XLU | mlp | 0.0341612965909243 | 0.4934196454561567 | 823 |
| y_XLU | ridge | 0.0342390483471221 | 0.4606843954278031 | 823 |
| y_XLV | baseline_zero | 0.0098573370333621 | 0.0036007130124777 | 823 |
| y_XLV | knn | 0.0100859631057425 | 0.4942944043988668 | 823 |
| y_XLV | rf | 0.0102046222312559 | 0.4141235515434298 | 823 |
| y_XLV | baseline_last | 0.0128651426763523 | 0.3794216179312528 | 823 |
| y_XLV | ridge | 0.0300956139157199 | 0.4765657569070348 | 823 |
| y_XLV | mlp | 0.0317832570492975 | 0.4727302224249485 | 823 |
| y_XLY | baseline_zero | 0.013606449757084 | 0.0 | 823 |
| y_XLY | rf | 0.01398375370753 | 0.5001904800601555 | 823 |
| y_XLY | knn | 0.0142308716102894 | 0.457805091846674 | 823 |
| y_XLY | baseline_last | 0.018868324146255 | 0.28946604401219 | 823 |
| y_XLY | mlp | 0.0349191325816634 | 0.4915335488935894 | 823 |
| y_XLY | ridge | 0.0488908931807354 | 0.4642067054739467 | 823 |

## Best model per target (RMSE)

| target | best_model_rmse |
| --- | --- |
| y_BTC-USD | baseline_zero |
| y_GLD | baseline_zero |
| y_TLT | baseline_zero |
| y_UUP | baseline_zero |
| y_XLB | baseline_zero |
| y_XLC | baseline_zero |
| y_XLE | baseline_zero |
| y_XLF | baseline_zero |
| y_XLI | baseline_zero |
| y_XLK | baseline_zero |
| y_XLP | baseline_zero |
| y_XLRE | baseline_zero |
| y_XLU | baseline_zero |
| y_XLV | baseline_zero |
| y_XLY | baseline_zero |

## Best model per target (directional_accuracy)

| target | best_model_directional_accuracy |
| --- | --- |
| y_BTC-USD | knn |
| y_GLD | rf |
| y_TLT | rf |
| y_UUP | rf |
| y_XLB | knn |
| y_XLC | rf |
| y_XLE | ridge |
| y_XLF | knn |
| y_XLI | knn |
| y_XLK | ridge |
| y_XLP | mlp |
| y_XLRE | rf |
| y_XLU | knn |
| y_XLV | knn |
| y_XLY | rf |

## Directional accuracy summary by target (gaps vs baselines)

| target | best_model_directional_accuracy | da_mean | da_std | da_baseline_last | da_baseline_zero | gap_vs_baseline_last | gap_vs_baseline_zero |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y_BTC-USD | knn | 0.4946666666666666 | 0.0504909232766973 | 0.4933333333333333 | 0.0 | 0.0013333333333332975 | 0.4946666666666666 |
| y_GLD | rf | 0.5094850551731891 | 0.0564566046147347 | 0.3187361858447051 | 0.0 | 0.190748869328484 | 0.5094850551731891 |
| y_TLT | rf | 0.5424672029053368 | 0.069886566280496 | 0.3473534052017824 | 0.0035314489432136 | 0.19511379770355441 | 0.5389357539621232 |
| y_UUP | rf | 0.5139594506191869 | 0.0995426070436737 | 0.3163699308750018 | 0.0286517418079284 | 0.1975895197441851 | 0.4853077088112585 |
| y_XLB | knn | 0.490097870340771 | 0.0970519055364066 | 0.3458560526557484 | 0.0119449961802903 | 0.1442418176850226 | 0.4781528741604807 |
| y_XLC | rf | 0.5531779060612732 | 0.0824123921960412 | 0.3893064368008587 | 0.0067335907335907 | 0.16387146926041446 | 0.5464443153276825 |
| y_XLE | ridge | 0.5109376985781651 | 0.1043812007656169 | 0.3608439582465952 | 0.011372264126828 | 0.15009374033156986 | 0.49956543445133705 |
| y_XLF | knn | 0.4816213436243862 | 0.0978929344390646 | 0.3241467854409031 | 0.010453781512605 | 0.1574745581834831 | 0.4711675621117812 |
| y_XLI | knn | 0.5077713642490519 | 0.0919460677980939 | 0.3046477700007111 | 0.0 | 0.2031235942483408 | 0.5077713642490519 |
| y_XLK | ridge | 0.495132721345196 | 0.0846278296907936 | 0.3493916258333094 | 0.0 | 0.14574109551188658 | 0.495132721345196 |
| y_XLP | mlp | 0.4987768662342698 | 0.0729809889778251 | 0.2976059781364041 | 0.0143265983509391 | 0.2011708880978657 | 0.48445026788333073 |
| y_XLRE | rf | 0.5293094051745166 | 0.0913611055894579 | 0.3057789906816276 | 0.020133933570039 | 0.22353041449288902 | 0.5091754716044776 |
| y_XLU | knn | 0.525855758856773 | 0.0755112343077721 | 0.3444115695540645 | 0.0087321878594698 | 0.18144418930270856 | 0.5171235709973032 |
| y_XLV | knn | 0.4942944043988668 | 0.0857791159818893 | 0.3794216179312528 | 0.0036007130124777 | 0.11487278646761401 | 0.4906936913863891 |
| y_XLY | rf | 0.5001904800601555 | 0.0902210476343854 | 0.28946604401219 | 0.0 | 0.21072443604796554 | 0.5001904800601555 |

## Automatic comment

- baseline_zero leads on global RMSE but not on directional_accuracy; this suggests returns concentrated near zero where predicting zero lowers average error without improving direction.

## Diagnostic output files

- `reports/metrics/target_distribution.csv`
- `reports/metrics/target_model_ranking.csv`
- `reports/metrics/best_model_by_target_rmse.csv`
- `reports/metrics/best_model_by_target_directional_accuracy.csv`
- `reports/metrics/directional_accuracy_significance.csv`
- `reports/metrics/directional_accuracy_ranking_by_target.csv`
- `reports/metrics/directional_accuracy_ranking_by_target_fold.csv`
- `reports/metrics/directional_accuracy_target_summary.csv`
- `reports/metrics/backtest_by_target.csv`
- `reports/metrics/backtest_global.csv`

## Plots

- `reports/plots/leaderboard_*.png`
- `reports/plots/targets_r2_comparison.png`
- `reports/plots/targets_directional_accuracy_comparison.png`
- `reports/plots/mlp/*_rolling_da.png`

## Minimal economic validation

- Transaction cost used: 1.0 bps per position change.

## Recommended final setup (Sharpe)

| cost_bps | recommended_scope | global_setup | global_sharpe | per_target_avg_best_sharpe | rationale |
| --- | --- | --- | --- | --- | --- |
| 1.0 | per-target | mlp \| thr=0.0 \| long_flat | 0.3693500585595554 | 1.1804036861345595 | Per-target beats average global Sharpe. |
| 3.0 | N/A | N/A | nan | nan | Insufficient data for this cost level. |
| 5.0 | N/A | N/A | nan | nan | Insufficient data for this cost level. |

### Recommendation by cost

- 1 bps: per-target
- 3 bps: N/A
- 5 bps: N/A

### Best global threshold by model

| scope | model | target | strategy | cost_bps | best_by | best_threshold | metric_value | metric_at_threshold0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global | baseline_last | ALL | long_flat | 1.0 | cum_return | 0.005 | 0.0231733892219621 | -0.0383010543690661 |
| global | baseline_last | ALL | long_flat | 1.0 | sharpe | 0.005 | -0.0007632747729399 | -0.3047260180877991 |
| global | baseline_last | ALL | long_short | 1.0 | cum_return | 0.005 | -0.2199023142674388 | -0.3779604403705573 |
| global | baseline_last | ALL | long_short | 1.0 | sharpe | 0.005 | -0.5231991319267568 | -0.9104811465415532 |
| global | baseline_zero | ALL | long_flat | 1.0 | cum_return | 0.0 | 0.0 | 0.0 |
| global | baseline_zero | ALL | long_flat | 1.0 | sharpe | 0.0 | nan | nan |
| global | baseline_zero | ALL | long_short | 1.0 | cum_return | 0.0 | 0.0 | 0.0 |
| global | baseline_zero | ALL | long_short | 1.0 | sharpe | 0.0 | nan | nan |
| global | knn | ALL | long_flat | 1.0 | cum_return | 0.0 | 0.0602893498461204 | 0.0602893498461204 |
| global | knn | ALL | long_flat | 1.0 | sharpe | 0.005 | 0.3676755708572429 | 0.1410228180035652 |
| global | knn | ALL | long_short | 1.0 | cum_return | 0.005 | -0.0749419785213554 | -0.2575513862663268 |
| global | knn | ALL | long_short | 1.0 | sharpe | 0.005 | -0.1657237647494036 | -0.3952121560632602 |
| global | mlp | ALL | long_flat | 1.0 | cum_return | 0.0 | 0.2167672833338296 | 0.2167672833338296 |
| global | mlp | ALL | long_flat | 1.0 | sharpe | 0.0 | 0.3693500585595554 | 0.3693500585595554 |
| global | mlp | ALL | long_short | 1.0 | cum_return | 0.005 | -0.0139976572839096 | -0.0491510366435402 |
| global | mlp | ALL | long_short | 1.0 | sharpe | 0.0 | -0.0510596926612911 | -0.0510596926612911 |
| global | rf | ALL | long_flat | 1.0 | cum_return | 0.0 | 0.2298042650519929 | 0.2298042650519929 |
| global | rf | ALL | long_flat | 1.0 | sharpe | 0.0 | 0.3305272730775217 | 0.3305272730775217 |
| global | rf | ALL | long_short | 1.0 | cum_return | 0.005 | 0.103693870621772 | 0.0331133715934595 |
| global | rf | ALL | long_short | 1.0 | sharpe | 0.005 | 0.2217365596661694 | -0.0794039526775565 |
| global | ridge | ALL | long_flat | 1.0 | cum_return | 0.0 | -0.062187424723027 | -0.062187424723027 |
| global | ridge | ALL | long_flat | 1.0 | sharpe | 0.0 | -0.2009494766602294 | -0.2009494766602294 |
| global | ridge | ALL | long_short | 1.0 | cum_return | 0.0005 | -0.3924838048808179 | -0.3972685642879315 |
| global | ridge | ALL | long_short | 1.0 | sharpe | 0.0005 | -0.8136273037271542 | -0.8245121891811077 |

### Best threshold per target

| scope | model | target | strategy | cost_bps | best_by | best_threshold | metric_value | metric_at_threshold0 | improves_vs_t0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| per_target | baseline_last | y_BTC-USD | long_flat | 1.0 | cum_return | 0.002 | 0.7623562185576858 | 0.662099148997408 | True |
| per_target | baseline_last | y_BTC-USD | long_flat | 1.0 | sharpe | 0.002 | 0.5210347683575777 | 0.4652775352831886 | True |
| per_target | baseline_last | y_BTC-USD | long_short | 1.0 | cum_return | 0.0 | -0.3989670616455779 | -0.3989670616455779 | False |
| per_target | baseline_last | y_BTC-USD | long_short | 1.0 | sharpe | 0.0 | -0.03452816641904 | -0.03452816641904 | False |
| per_target | baseline_last | y_GLD | long_flat | 1.0 | cum_return | 0.003 | 0.6038900862583316 | 0.1720786661376465 | True |
| per_target | baseline_last | y_GLD | long_flat | 1.0 | sharpe | 0.005 | 1.3863184437979217 | 0.4114014938133222 | True |
| per_target | baseline_last | y_GLD | long_short | 1.0 | cum_return | 0.005 | 0.394220889720452 | -0.2996298525556092 | True |
| per_target | baseline_last | y_GLD | long_short | 1.0 | sharpe | 0.005 | 0.8547158536886904 | -0.5485044512455958 | True |
| per_target | baseline_last | y_TLT | long_flat | 1.0 | cum_return | 0.003 | -0.1048961038102357 | -0.289436530519552 | True |
| per_target | baseline_last | y_TLT | long_flat | 1.0 | sharpe | 0.003 | -0.5026890946621853 | -1.3762791773516083 | True |
| per_target | baseline_last | y_TLT | long_short | 1.0 | cum_return | 0.003 | 0.0227808870487919 | -0.3393357299822497 | True |
| per_target | baseline_last | y_TLT | long_short | 1.0 | sharpe | 0.003 | 0.1228322690177964 | -1.1650840876694823 | True |
| per_target | baseline_last | y_UUP | long_flat | 1.0 | cum_return | 0.002 | 0.1062405016543779 | -0.0926279896432135 | True |
| per_target | baseline_last | y_UUP | long_flat | 1.0 | sharpe | 0.002 | 0.862511891170455 | -0.5913568213344857 | True |
| per_target | baseline_last | y_UUP | long_short | 1.0 | cum_return | 0.002 | 0.0145659783968592 | -0.2825513008724142 | True |
| per_target | baseline_last | y_UUP | long_short | 1.0 | sharpe | 0.002 | 0.1109128822427326 | -1.5584263223859651 | True |
| per_target | baseline_last | y_XLB | long_flat | 1.0 | cum_return | 0.005 | -0.2195224057567097 | -0.2278235169334694 | True |
| per_target | baseline_last | y_XLB | long_flat | 1.0 | sharpe | 0.0 | -0.7250700192428444 | -0.7250700192428444 | False |
| per_target | baseline_last | y_XLB | long_short | 1.0 | cum_return | 0.005 | -0.4477450645733403 | -0.4550311938687859 | True |
| per_target | baseline_last | y_XLB | long_short | 1.0 | sharpe | 0.0 | -0.9507572267259464 | -0.9507572267259464 | False |
| per_target | baseline_last | y_XLC | long_flat | 1.0 | cum_return | 0.005 | 0.1983405847724462 | -0.0639817955295756 | True |
| per_target | baseline_last | y_XLC | long_flat | 1.0 | sharpe | 0.005 | 0.8426921877955976 | -0.1589867416908054 | True |
| per_target | baseline_last | y_XLC | long_short | 1.0 | cum_return | 0.005 | -0.0923912725221053 | -0.2214476446965073 | True |
| per_target | baseline_last | y_XLC | long_short | 1.0 | sharpe | 0.005 | -0.1473797246997392 | -0.378398251814667 | True |
| per_target | baseline_last | y_XLE | long_flat | 1.0 | cum_return | 0.002 | 0.521646094801266 | 0.2684254956230287 | True |
| per_target | baseline_last | y_XLE | long_flat | 1.0 | sharpe | 0.002 | 1.066686889634897 | 0.5576523205738567 | True |
| per_target | baseline_last | y_XLE | long_short | 1.0 | cum_return | 0.002 | 0.2207923415975539 | -0.0817932987842909 | True |
| per_target | baseline_last | y_XLE | long_short | 1.0 | sharpe | 0.002 | 0.406052392708543 | -0.00873710578461 | True |
| per_target | baseline_last | y_XLF | long_flat | 1.0 | cum_return | 0.005 | 0.1860508922294139 | -0.0375011946013583 | True |
| per_target | baseline_last | y_XLF | long_flat | 1.0 | sharpe | 0.005 | 0.561941437303436 | -0.0473642197021883 | True |
| per_target | baseline_last | y_XLF | long_short | 1.0 | cum_return | 0.005 | 0.4333437961955177 | -0.5091164256115621 | True |
| per_target | baseline_last | y_XLF | long_short | 1.0 | sharpe | 0.005 | 0.8520546187127971 | -1.1413346848851489 | True |
| per_target | baseline_last | y_XLI | long_flat | 1.0 | cum_return | 0.005 | 0.1274085962880244 | -0.1497435536785252 | True |
| per_target | baseline_last | y_XLI | long_flat | 1.0 | sharpe | 0.005 | 0.4448174829110367 | -0.4200186889554793 | True |
| per_target | baseline_last | y_XLI | long_short | 1.0 | cum_return | 0.005 | -0.2766036316892476 | -0.5645896769649199 | True |
| per_target | baseline_last | y_XLI | long_short | 1.0 | sharpe | 0.005 | -0.5422023560275118 | -1.370196949928777 | True |
| per_target | baseline_last | y_XLK | long_flat | 1.0 | cum_return | 0.005 | -0.137808868685242 | -0.244106862031972 | True |
| per_target | baseline_last | y_XLK | long_flat | 1.0 | sharpe | 0.005 | -0.2747292175342855 | -0.4774893447516971 | True |
| per_target | baseline_last | y_XLK | long_short | 1.0 | cum_return | 0.001 | -0.4941013201429538 | -0.5330142007450225 | True |
| per_target | baseline_last | y_XLK | long_short | 1.0 | sharpe | 0.001 | -0.724002445468265 | -0.8168357927782068 | True |
| per_target | baseline_last | y_XLP | long_flat | 1.0 | cum_return | 0.003 | -0.1288020941507081 | -0.2169586931795197 | True |
| per_target | baseline_last | y_XLP | long_flat | 1.0 | sharpe | 0.003 | -0.5510637930438088 | -0.928598078880402 | True |
| per_target | baseline_last | y_XLP | long_short | 1.0 | cum_return | 0.005 | -0.3480089606281649 | -0.5373618219916099 | True |
| per_target | baseline_last | y_XLP | long_short | 1.0 | sharpe | 0.005 | -1.2861702628542662 | -1.9773264516771671 | True |
| per_target | baseline_last | y_XLRE | long_flat | 1.0 | cum_return | 0.005 | -0.1969445017744733 | -0.2887940764113619 | True |
| per_target | baseline_last | y_XLRE | long_flat | 1.0 | sharpe | 0.005 | -1.012966658052464 | -1.2116407669184712 | True |
| per_target | baseline_last | y_XLRE | long_short | 1.0 | cum_return | 0.005 | -0.3992317027319071 | -0.5375743394839573 | True |
| per_target | baseline_last | y_XLRE | long_short | 1.0 | sharpe | 0.005 | -1.252676457209096 | -1.567182024469937 | True |
| per_target | baseline_last | y_XLU | long_flat | 1.0 | cum_return | 0.0 | 0.3148312469512473 | 0.3148312469512473 | False |
| per_target | baseline_last | y_XLU | long_flat | 1.0 | sharpe | 0.0 | 0.9144304650105124 | 0.9144304650105124 | False |
| per_target | baseline_last | y_XLU | long_short | 1.0 | cum_return | 0.005 | -0.0787412164298868 | -0.2003319818219002 | True |
| per_target | baseline_last | y_XLU | long_short | 1.0 | sharpe | 0.005 | -0.194731738095407 | -0.4138983304571245 | True |
| per_target | baseline_last | y_XLV | long_flat | 1.0 | cum_return | 0.003 | 0.0919457708292958 | 0.0427012801780986 | True |
| per_target | baseline_last | y_XLV | long_flat | 1.0 | sharpe | 0.003 | 0.3931376489180341 | 0.1973818094034185 | True |
| per_target | baseline_last | y_XLV | long_short | 1.0 | cum_return | 0.0005 | 0.0602332290600649 | 0.0480456196021497 | True |
| per_target | baseline_last | y_XLV | long_short | 1.0 | sharpe | 0.0005 | 0.1975009971612029 | 0.1723037969920574 | True |
| per_target | baseline_last | y_XLY | long_flat | 1.0 | cum_return | 0.005 | -0.1814238005759711 | -0.423677440894873 | True |
| per_target | baseline_last | y_XLY | long_flat | 1.0 | sharpe | 0.005 | -0.4914987823071416 | -1.1802300365733034 | True |
| per_target | baseline_last | y_XLY | long_short | 1.0 | cum_return | 0.002 | -0.6499774829845868 | -0.7567076961361021 | True |
| per_target | baseline_last | y_XLY | long_short | 1.0 | sharpe | 0.002 | -1.472549457730291 | -1.8983111488736888 | True |
| per_target | baseline_zero | y_BTC-USD | long_flat | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_BTC-USD | long_flat | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_BTC-USD | long_short | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_BTC-USD | long_short | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_GLD | long_flat | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_GLD | long_flat | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_GLD | long_short | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_GLD | long_short | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_TLT | long_flat | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_TLT | long_flat | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_TLT | long_short | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_TLT | long_short | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_UUP | long_flat | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_UUP | long_flat | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_UUP | long_short | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_UUP | long_short | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLB | long_flat | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLB | long_flat | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLB | long_short | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLB | long_short | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLC | long_flat | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLC | long_flat | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLC | long_short | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLC | long_short | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLE | long_flat | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLE | long_flat | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLE | long_short | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLE | long_short | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLF | long_flat | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLF | long_flat | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLF | long_short | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLF | long_short | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLI | long_flat | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLI | long_flat | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLI | long_short | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLI | long_short | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLK | long_flat | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLK | long_flat | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLK | long_short | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLK | long_short | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLP | long_flat | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLP | long_flat | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLP | long_short | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLP | long_short | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLRE | long_flat | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLRE | long_flat | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLRE | long_short | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLRE | long_short | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLU | long_flat | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLU | long_flat | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLU | long_short | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLU | long_short | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLV | long_flat | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLV | long_flat | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLV | long_short | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLV | long_short | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLY | long_flat | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLY | long_flat | 1.0 | sharpe | 0.0 | nan | nan | False |
| per_target | baseline_zero | y_XLY | long_short | 1.0 | cum_return | 0.0 | 0.0 | 0.0 | False |
| per_target | baseline_zero | y_XLY | long_short | 1.0 | sharpe | 0.0 | nan | nan | False |

### Comparison vs threshold=0

| targets_improve_cum_return_vs_t0 | targets_improve_sharpe_vs_t0 |
| --- | --- |
| 109/180 | 113/180 |

### Global backtest table

| model | strategy | threshold | cost_bps | n_targets | avg_cum_return | avg_sharpe | avg_hit_rate | avg_active_rate | avg_mean_ret | avg_std_ret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | long_flat | 0.0 | 1.0 | 15 | 0.2167672833338296 | 0.3693500585595554 | 0.2690371270419873 | 0.5085370055353045 | 0.0002076343705316 | 0.0085444652512157 |
| knn | long_flat | 0.005 | 1.0 | 15 | -0.0161121442966901 | 0.3676755708572429 | 0.0239227757526663 | 0.0309027136492507 | -7.176377903140372e-06 | 0.0016566386931278 |
| mlp | long_flat | 0.001 | 1.0 | 15 | 0.21072424310238 | 0.360965838502704 | 0.2583782368030241 | 0.4876973673552046 | 0.0001980863511125 | 0.0083646301816477 |
| mlp | long_flat | 0.0005 | 1.0 | 15 | 0.1864826145660933 | 0.3433369992649821 | 0.2634083974618604 | 0.4984007020386121 | 0.0001856903091896 | 0.0084309302090993 |
| rf | long_flat | 0.0 | 1.0 | 15 | 0.2298042650519929 | 0.3305272730775217 | 0.2922452274875118 | 0.5590815714864318 | 0.0001616876314148 | 0.0087594519879372 |
| mlp | long_flat | 0.002 | 1.0 | 15 | 0.194743193297045 | 0.3164719132754111 | 0.2480354799513973 | 0.4680857027136492 | 0.000172129742114 | 0.0082263829028927 |
| rf | long_flat | 0.0005 | 1.0 | 15 | 0.175826201151677 | 0.3044391899253355 | 0.2443343323882813 | 0.4606755231537734 | 0.0001370475069128 | 0.0079003231902384 |
| mlp | long_flat | 0.003 | 1.0 | 15 | 0.1982563537522355 | 0.2868416434513034 | 0.2379228297556365 | 0.4487851491832051 | 0.0001623583528702 | 0.0080957272434684 |
| knn | long_flat | 0.0005 | 1.0 | 15 | 0.0537186431600129 | 0.2001145288865967 | 0.1652346159038747 | 0.3032101525583907 | 6.473972628351158e-05 | 0.0060828550542788 |
| mlp | long_flat | 0.005 | 1.0 | 15 | 0.1447333899509459 | 0.1920412039645292 | 0.217297529364115 | 0.4095281220467126 | 0.0001074642341888 | 0.007607045728278 |
| rf | long_flat | 0.003 | 1.0 | 15 | 0.1724425955609041 | 0.1896504473017015 | 0.0707805589307411 | 0.1215057108140947 | 8.304209922181549e-05 | 0.0041366657857646 |
| knn | long_flat | 0.001 | 1.0 | 15 | 0.0508310505938027 | 0.1664163607303487 | 0.1307604158228702 | 0.2374782773052518 | 5.417077116416362e-05 | 0.0054245106848155 |
| knn | long_flat | 0.0 | 1.0 | 15 | 0.0602893498461204 | 0.1410228180035652 | 0.1989826380450924 | 0.3701363034966923 | 6.345564971208716e-05 | 0.0066379838039925 |
| rf | long_flat | 0.001 | 1.0 | 15 | 0.1078630185106799 | 0.124646033626554 | 0.1934183880113406 | 0.3650444174429593 | 7.268709547095367e-05 | 0.0071679450674735 |
| rf | long_flat | 0.002 | 1.0 | 15 | 0.0900609079342583 | 0.1031629055812817 | 0.1179652760901849 | 0.2170751451329823 | 5.579324598582108e-05 | 0.0054387858904556 |
| rf | long_flat | 0.005 | 1.0 | 15 | 0.0981590863854534 | 0.0267543681495939 | 0.0302554340488726 | 0.0426841636289996 | 3.8570307101140814e-05 | 0.0022106424483711 |
| baseline_last | long_flat | 0.005 | 1.0 | 15 | 0.0231733892219621 | -0.0007632747729399 | 0.1217036857027136 | 0.217852247873633 | 2.5914989131559065e-05 | 0.0057728314952864 |
| baseline_last | long_flat | 0.003 | 1.0 | 15 | 0.0118913190868562 | -0.0517011252556964 | 0.1516394761711894 | 0.2786069933846361 | -8.7251298403775e-06 | 0.0063392803650709 |
| baseline_last | long_flat | 0.002 | 1.0 | 15 | 0.013308002412056 | -0.1355259560381315 | 0.1686353449439719 | 0.315423140272715 | -2.6244619699959748e-05 | 0.0067350601120974 |
| ridge | long_flat | 0.0 | 1.0 | 15 | -0.062187424723027 | -0.2009494766602294 | 0.2525595247738625 | 0.496771270419873 | -5.876712455744004e-05 | 0.0088083759941415 |
| ridge | long_flat | 0.0005 | 1.0 | 15 | -0.0653542600939513 | -0.2033206035057828 | 0.2462383151073309 | 0.4838127176994735 | -6.198873620192042e-05 | 0.0086665777956223 |
| knn | long_flat | 0.002 | 1.0 | 15 | -0.0423669982932819 | -0.2198165539430203 | 0.0748666396651815 | 0.1381625489401917 | -5.430198543761671e-05 | 0.00419165018078 |
| baseline_last | long_flat | 0.001 | 1.0 | 15 | -0.0188962743998259 | -0.2339313780649558 | 0.1828089644930471 | 0.3480765222087215 | -5.036031830438135e-05 | 0.0070615581052028 |
| baseline_last | long_flat | 0.0005 | 1.0 | 15 | -0.0316368735609608 | -0.2516538562540399 | 0.1900656676117186 | 0.364408586472256 | -6.49145918738605e-05 | 0.0071500126943647 |
| ridge | long_flat | 0.001 | 1.0 | 15 | -0.090669174830773 | -0.2551790485303667 | 0.2398361009855542 | 0.4727903874713109 | -9.407823710803851e-05 | 0.0085653603414407 |
| baseline_last | long_flat | 0.0 | 1.0 | 15 | -0.0383010543690661 | -0.3047260180877991 | 0.194973241528284 | 0.3773542324827865 | -8.219244528103915e-05 | 0.0071936464367232 |
| knn | long_flat | 0.003 | 1.0 | 15 | -0.0530499848237469 | -0.3262398812149145 | 0.0479380585932226 | 0.0822465775617658 | -5.324098927171755e-05 | 0.0031071771689065 |
| ridge | long_flat | 0.002 | 1.0 | 15 | -0.1112926156996956 | -0.3304682250471585 | 0.2270553260429323 | 0.4473621709194005 | -0.000123311766112 | 0.0083492467732186 |
| ridge | long_flat | 0.003 | 1.0 | 15 | -0.1334546517016804 | -0.3870852857870165 | 0.2148860267314702 | 0.4239067368705279 | -0.0001517318159355 | 0.0080780909564655 |
| ridge | long_flat | 0.005 | 1.0 | 15 | -0.1742078488737513 | -0.5325330686018503 | 0.1916943971918455 | 0.379308437964088 | -0.0002117697585736 | 0.007667311539939 |
| baseline_zero | long_flat | 0.0 | 1.0 | 15 | 0.0 | nan | 0.0080194410692588 | 0.0 | 0.0 | 0.0 |
| baseline_zero | long_flat | 0.0005 | 1.0 | 15 | 0.0 | nan | 0.0080194410692588 | 0.0 | 0.0 | 0.0 |
| baseline_zero | long_flat | 0.001 | 1.0 | 15 | 0.0 | nan | 0.0080194410692588 | 0.0 | 0.0 | 0.0 |
| baseline_zero | long_flat | 0.002 | 1.0 | 15 | 0.0 | nan | 0.0080194410692588 | 0.0 | 0.0 | 0.0 |
| baseline_zero | long_flat | 0.003 | 1.0 | 15 | 0.0 | nan | 0.0080194410692588 | 0.0 | 0.0 | 0.0 |
| baseline_zero | long_flat | 0.005 | 1.0 | 15 | 0.0 | nan | 0.0080194410692588 | 0.0 | 0.0 | 0.0 |
| rf | long_short | 0.005 | 1.0 | 15 | 0.103693870621772 | 0.2217365596661694 | 0.0417573106520858 | 0.0669452679897394 | 6.251258205630062e-05 | 0.0033493794480289 |
| rf | long_short | 0.003 | 1.0 | 15 | 0.0968545437645859 | 0.0754453981793364 | 0.1048761981909005 | 0.2017723774807614 | 3.959847133105928e-05 | 0.0056359671485357 |
| rf | long_short | 0.0005 | 1.0 | 15 | 0.0375492783084857 | -0.0088022149016439 | 0.3982068313757256 | 0.8080926150938301 | -7.03765698575941e-05 | 0.0110571293680735 |
| mlp | long_short | 0.0 | 1.0 | 15 | -0.0491510366435402 | -0.0510596926612911 | 0.4902551100310517 | 1.0 | -4.462143349948664e-05 | 0.0122126012167652 |
| rf | long_short | 0.0 | 1.0 | 15 | 0.0331133715934595 | -0.0794039526775565 | 0.4875848251653841 | 1.0 | -0.0001365075997309 | 0.0121884045560788 |
| mlp | long_short | 0.001 | 1.0 | 15 | -0.0669219638074496 | -0.0860220107035407 | 0.4692979073849062 | 0.9579078169299312 | -6.097332686872787e-05 | 0.0119477458210446 |
| mlp | long_short | 0.0005 | 1.0 | 15 | -0.078948663461248 | -0.0893212134408679 | 0.4796248953692453 | 0.9792334818415012 | -7.420970543790622e-05 | 0.0120602854132135 |
| rf | long_short | 0.002 | 1.0 | 15 | -0.0073043079573869 | -0.099005114360445 | 0.1810038612123667 | 0.3635582557040637 | -3.7591403475752456e-05 | 0.0074454937498476 |
| mlp | long_short | 0.003 | 1.0 | 15 | -0.0296946480895792 | -0.1030337292176667 | 0.4309269069798839 | 0.8784761981909005 | -4.568009957405017e-05 | 0.0114072430371957 |
| rf | long_short | 0.001 | 1.0 | 15 | -0.0114202156655614 | -0.1089073261285762 | 0.3090037532064263 | 0.6310427433508843 | -9.091684090564454e-05 | 0.0098087858255915 |
| mlp | long_short | 0.002 | 1.0 | 15 | -0.0308540738478466 | -0.1116807787331312 | 0.4494539759686783 | 0.917780531929256 | -6.091610819796971e-05 | 0.0116765348847652 |
| mlp | long_short | 0.005 | 1.0 | 15 | -0.0139976572839096 | -0.1202975075240806 | 0.3932405022276225 | 0.7993062238423113 | -4.583721842905948e-05 | 0.0107272941648569 |
| knn | long_short | 0.005 | 1.0 | 15 | -0.0749419785213554 | -0.1657237647494036 | 0.074805616308897 | 0.1328454705008775 | -0.000104141585846 | 0.0047991643844262 |
| knn | long_short | 0.001 | 1.0 | 15 | -0.2389147151132139 | -0.3699035087932172 | 0.3609397596867827 | 0.7328703118671527 | -0.0002985279863916 | 0.0109163363190864 |
| knn | long_short | 0.0 | 1.0 | 15 | -0.2575513862663268 | -0.3952121560632602 | 0.4851446469555825 | 1.0 | -0.0003330849693738 | 0.0122139429569867 |
| knn | long_short | 0.0005 | 1.0 | 15 | -0.2561516132186148 | -0.4011436337903766 | 0.4243956257594167 | 0.8657584447144593 | -0.0003230948471057 | 0.0115924761528809 |
| baseline_last | long_short | 0.005 | 1.0 | 15 | -0.2199023142674388 | -0.5231991319267568 | 0.2089668691778047 | 0.4225760496827325 | -0.0003170258473744 | 0.0093086353677649 |
| knn | long_short | 0.003 | 1.0 | 15 | -0.2133343431910895 | -0.5264588862540495 | 0.1698653705953827 | 0.3341966248143648 | -0.0002915012800847 | 0.0079140084852492 |
| knn | long_short | 0.002 | 1.0 | 15 | -0.2632666448799428 | -0.6114925349915313 | 0.2500590252463885 | 0.5063840691238018 | -0.0003475956714857 | 0.0094170203385424 |
| baseline_last | long_short | 0.003 | 1.0 | 15 | -0.2961263282431451 | -0.647042719626565 | 0.2610046712569191 | 0.5400539489671932 | -0.0004212423329035 | 0.0101252476212074 |
| baseline_last | long_short | 0.002 | 1.0 | 15 | -0.3235097197257349 | -0.7099613343605579 | 0.2924587552315377 | 0.6079005265289591 | -0.0004531461398141 | 0.010515931742199 |
| baseline_last | long_short | 0.001 | 1.0 | 15 | -0.3618904568775069 | -0.8052305123885386 | 0.3187729985149183 | 0.6697527744025922 | -0.0004896526090468 | 0.0108763722447393 |
| ridge | long_short | 0.0005 | 1.0 | 15 | -0.3924838048808179 | -0.8136273037271542 | 0.4550411232617793 | 0.9735552855407048 | -0.0005705422551448 | 0.0120315326607297 |
| ridge | long_short | 0.0 | 1.0 | 15 | -0.3972685642879315 | -0.8245121891811077 | 0.4668785203186175 | 1.0 | -0.0005774568254598 | 0.0122026826693281 |
| ridge | long_short | 0.001 | 1.0 | 15 | -0.3982125425268533 | -0.8396205642166424 | 0.4428116106385851 | 0.9497808559470772 | -0.0005880210688041 | 0.0118536566909616 |
| baseline_last | long_short | 0.0005 | 1.0 | 15 | -0.3822567020973882 | -0.9011915825908166 | 0.3322283245578507 | 0.7047244498447415 | -0.0005458345665562 | 0.0110130992523029 |
| baseline_last | long_short | 0.0 | 1.0 | 15 | -0.3779604403705573 | -0.9104811465415532 | 0.3451610638585122 | 0.7327663021466181 | -0.0005564265665495 | 0.011087886413535 |
| ridge | long_short | 0.003 | 1.0 | 15 | -0.4145946291949753 | -0.9359837578888432 | 0.3969122181719994 | 0.8510235722964763 | -0.0006281418673136 | 0.0112254351685195 |
| ridge | long_short | 0.005 | 1.0 | 15 | -0.3988066658331318 | -0.9390334544951516 | 0.3548042662346429 | 0.7615058188200351 | -0.0006053739417736 | 0.0106464165787827 |
| ridge | long_short | 0.002 | 1.0 | 15 | -0.4189786316896565 | -0.9425301194497492 | 0.4182092075064129 | 0.8977408937491561 | -0.0006361680086456 | 0.0115400746958118 |
| baseline_zero | long_short | 0.0 | 1.0 | 15 | 0.0 | nan | 0.0080194410692588 | 0.0 | 0.0 | 0.0 |
| baseline_zero | long_short | 0.0005 | 1.0 | 15 | 0.0 | nan | 0.0080194410692588 | 0.0 | 0.0 | 0.0 |
| baseline_zero | long_short | 0.001 | 1.0 | 15 | 0.0 | nan | 0.0080194410692588 | 0.0 | 0.0 | 0.0 |
| baseline_zero | long_short | 0.002 | 1.0 | 15 | 0.0 | nan | 0.0080194410692588 | 0.0 | 0.0 | 0.0 |
| baseline_zero | long_short | 0.003 | 1.0 | 15 | 0.0 | nan | 0.0080194410692588 | 0.0 | 0.0 | 0.0 |
| baseline_zero | long_short | 0.005 | 1.0 | 15 | 0.0 | nan | 0.0080194410692588 | 0.0 | 0.0 | 0.0 |