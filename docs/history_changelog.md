# History of Changelog

- :boom: **Jan 31, 2024**. Add `qalign` for both NR and IAA. It is our most powerful unified metric based on large vision-language models, and shows remarkable performance and robustness. Refer [Q-Align](https://github.com/Q-Future/Q-Align) for more details. Use it with the following codes:
  ```
  qalign = create_metric('qalign').cuda()
  quality_score = qalign(input, task_='quality')
  aesthetic_score = qalign(input, task_='aesthetic')
  ```
- **Jan 19, 2024**. Add `wadiqam_fr` and `wadiqam_nr`. All implemented methods are usable now 🍻. 
- **Dec 23, 2023**. Add `liqe` and `liqe_mix`. Thanks for the contribution from [Weixia Zhang](https://github.com/zwx8981) 🤗.
- **Oct 09, 2023**. Add datasets: [PIQ2023](https://github.com/DXOMARK-Research/PIQ2023), [GFIQA](http://database.mmsp-kn.de/gfiqa-20k-database.html). Add metric `topiq_nr-face`. We release example results on FFHQ [here](tests/ffhq_score_topiq_nr-face.csv) for reference.
- **Aug 15, 2023**. Add `st-lpips` and `laion_aes`. Refer to official repo at [ShiftTolerant-LPIPS](https://github.com/abhijay9/ShiftTolerant-LPIPS) and [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
- **Aug 05, 2023**. Add our work [TOPIQ](https://arxiv.org/abs/2308.03060) with remarkable performance on almost all benchmarks via efficient Resnet50 backbone. Use it with `topiq_fr, topiq_nr, topiq_iaa` for Full-Reference, No-Reference and Aesthetic assessment respectively.
- **March 30, 2023**. Add [URanker](https://github.com/RQ-Wu/UnderwaterRanker) for IQA of under water images. 
- **March 29, 2023**. :rotating_light: Hot fix of NRQM & PI. 
- **March 25, 2023**. Add TreS, HyperIQA, CNNIQA, CLIPIQA.
- **Sep 1, 2022**. 1) Add pretrained models for MANIQA and AHIQ. 2) Add dataset interface for pieapp and PIPAL.
- **June 3, 2022**. Add FID metric. See [clean-fid](https://github.com/GaParmar/clean-fid) for more details.
- **March 11, 2022**. Add pretrained DBCNN, NIMA, and official model of PieAPP, paq2piq.
- **March 5, 2022**. Add NRQM, PI, ILNIQE metrics.
- **Feb 2, 2022**. Add MUSIQ inference code, and the converted official weights. See [Official codes](https://github.com/google-research/google-research/tree/master/musiq).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=chaofengc/IQA-PyTorch&type=Date)](https://star-history.com/#chaofengc/IQA-PyTorch&Date)
