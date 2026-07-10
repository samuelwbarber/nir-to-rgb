# Bibliography Verification

Audit date: 2026-06-09

## Summary

- Bibliography entries checked: 49
- Entries verified as real: 49
- Entries that could not be verified: 0
- Entries needing a factual metadata correction: 1
- Software entry with a minor official-metadata discrepancy: 1
- Entries cited in the report: 42
- Entries present in `references.bib` but not cited: 7

An entry marked **Verified** has matching core metadata: title, authors, publication
or repository, and year. Missing optional fields such as DOI, page range, month, or
ISBN are noted only when useful; their absence does not make a reference false.

## Corrections

1. `Bashkatov_2005`
   - The paper is real and the DOI is correct.
   - Change `pages = {2543}` to `pages = {2543--2555}`.
   - Source: https://doi.org/10.1088/0022-3727/38/15/004

2. `jocher2023yolov8`
   - The software and version are real. Version 8.0.0 was released on 2023-01-10.
   - The official CFF software title is `Ultralytics YOLO`, and its author order is
     Glenn Jocher, Jing Qiu, Ayush Chaurasia. The current entry says
     `Ultralytics YOLOv8` and orders Chaurasia before Qiu.
   - This is a minor citation-normalization issue, not a fabricated reference.
   - Source: https://raw.githubusercontent.com/ultralytics/ultralytics/main/CITATION.cff

## Verified Entries

| BibTeX key | Status | Verification source / note |
|---|---|---|
| `deng2020retinaface` | Verified | CVF confirms authors, title, CVPR 2020, pp. 5203-5212: https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html |
| `jocher2023yolov8` | Verified; minor normalization | Official CFF confirms software, authors, version 8.0.0, and 2023 release: https://raw.githubusercontent.com/ultralytics/ultralytics/main/CITATION.cff |
| `tucker1979red` | Verified | DOI record: https://doi.org/10.1016/0034-4257(79)90013-0 |
| `he2011single` | Verified | DOI record: https://doi.org/10.1109/TPAMI.2010.168 |
| `curran1989remote` | Verified | DOI record: https://doi.org/10.1016/0034-4257(89)90069-2 |
| `gao1996ndwi` | Verified | DOI record: https://doi.org/10.1016/S0034-4257(96)00067-3 |
| `salamati2010material` | Verified | DOI record: https://doi.org/10.2352/CIC.2010.18.1.art00034 |
| `salamati2014incorporating` | Verified | arXiv record confirms title, four authors, and 2014: https://arxiv.org/abs/1406.6147 |
| `peng2018rgbnir` | Verified | DOI record: https://doi.org/10.1186/s13640-018-0388-1 |
| `krizhevsky2012imagenet` | Verified | Official NeurIPS record: https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html |
| `gonzalez2002digital` | Verified | Catalog record confirms authors, second edition, Prentice Hall, 2002: https://books.google.com/books/about/Digital_Image_Processing.html?id=wvRMPgAACAAJ |
| `efros1999texture` | Verified | DOI record: https://doi.org/10.1109/ICCV.1999.790383 |
| `dong2015image` | Verified | DOI record confirms the final journal publication is 2016, as stated in the entry: https://doi.org/10.1109/TPAMI.2015.2439281 |
| `Monno2019RGBNIR` | Verified | DOI record: https://doi.org/10.1109/JSEN.2018.2876774 |
| `s17061297` | Verified | DOI record: https://doi.org/10.3390/s17061297 |
| `li2007illumination` | Verified | Matching DOI is 10.1109/TPAMI.2007.1014: https://doi.org/10.1109/TPAMI.2007.1014 |
| `Bashkatov_2005` | Verified; correction needed | DOI record confirms pp. 2543-2555: https://doi.org/10.1088/0022-3727/38/15/004 |
| `miura2004feature` | Verified | Matching DOI is 10.1007/s00138-004-0149-2: https://doi.org/10.1007/s00138-004-0149-2 |
| `wang2018pix2pixhd` | Verified | CVF confirms CVPR 2018, pp. 8798-8807: https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_High-Resolution_Image_Synthesis_CVPR_2018_paper.html |
| `pix2next2024` | Verified | arXiv confirms title, six authors, and 2024: https://arxiv.org/abs/2409.16706 |
| `leonard2013survey` | Verified | DOI record confirms the journal publication year is 2014, as stated in the entry: https://doi.org/10.3934/dcds.2014.34.1533 |
| `debortoli2021neural` | Verified | Official NeurIPS record: https://proceedings.neurips.cc/paper_files/paper/2021/hash/940392f5f32a7ade1cc201767cf83e31-Abstract.html |
| `wang2023internimage` | Verified | CVF confirms authors, CVPR 2023, pp. 14408-14419: https://openaccess.thecvf.com/content/CVPR2023/html/Wang_InternImage_Exploring_Large-Scale_Vision_Foundation_Models_With_Deformable_Convolutions_CVPR_2023_paper.html |
| `zhang2000flexible` | Verified | DOI record: https://doi.org/10.1109/34.888718 |
| `hartley2004multiple` | Verified | Cambridge confirms authors and 2004 print publication: https://www.cambridge.org/core/books/multiple-view-geometry-in-computer-vision/contents/E40FB8D5492A980260E866775C0C044F |
| `epfl_rgbnir_dataset` | Verified | The exact URL in the entry is live and describes the 477-image dataset: https://ivrlwww.epfl.ch/supplementary_material/cvpr11/index.html |
| `brown2011multispectral` | Verified | DOI record: https://doi.org/10.1109/CVPR.2011.5995637 |
| `hwang2015multispectral` | Verified | DOI record: https://doi.org/10.1109/CVPR.2015.7298706 |
| `isola2017image` | Verified | CVF confirms CVPR 2017, pp. 1125-1134: https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html |
| `park2019spade` | Verified | CVF confirms CVPR 2019, pp. 2337-2346: https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Semantic_Image_Synthesis_With_Spatially-Adaptive_Normalization_CVPR_2019_paper.html |
| `howard2019mobilenetv3` | Verified | CVF confirms ICCV 2019, pp. 1314-1324: https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html |
| `goodfellow2014generative` | Verified | Official NeurIPS record: https://proceedings.neurips.cc/paper/5423-generative-adversarial-nets |
| `johnson2016perceptual` | Verified | Springer DOI record, pp. 694-711: https://doi.org/10.1007/978-3-319-46475-6_43 |
| `ho2020denoising` | Verified | Official NeurIPS record: https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html |
| `saharia2022image` | Verified | DOI is correct; final issue citation is 2023, vol. 45(4), pp. 4713-4726: https://doi.org/10.1109/TPAMI.2022.3204461 |
| `li2023bbdm` | Verified | CVF confirms CVPR 2023, pp. 1952-1961: https://openaccess.thecvf.com/content/CVPR2023/html/Li_BBDM_Image-to-Image_Translation_With_Brownian_Bridge_Diffusion_Models_CVPR_2023_paper.html |
| `chen2022nafnet` | Verified | Springer confirms ECCV 2022, pp. 17-33: https://doi.org/10.1007/978-3-031-20071-7_2 |
| `hinton2015distilling` | Verified | arXiv record confirms title, authors, and 2015: https://arxiv.org/abs/1503.02531 |
| `romero2015fitnets` | Verified | arXiv record confirms title and six authors; the work appeared at ICLR 2015: https://arxiv.org/abs/1412.6550 |
| `jacob2018quantization` | Verified | DOI record confirms CVPR 2018, pp. 2704-2713: https://doi.org/10.1109/CVPR.2018.00286 |
| `gholami2021survey` | Verified | arXiv record confirms title, six authors, and 2021: https://arxiv.org/abs/2103.13630 |
| `wang2003msssim` | Verified | DOI record confirms conference, volume 2, pp. 1398-1402: https://doi.org/10.1109/ACSSC.2003.1292216 |
| `zhang2018lpips` | Verified | CVF confirms CVPR 2018, pp. 586-595: https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html |
| `lugaresi2019mediapipe` | Verified | arXiv record confirms title and 2019. The BibTeX deliberately abbreviates the full author list with `others`: https://arxiv.org/abs/1906.08172 |
| `xie2021segformer` | Verified | Official NeurIPS record: https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html |
| `bao2022beit` | Verified | Official OpenReview record confirms ICLR 2022 and the four authors: https://openreview.net/forum?id=p-BhZSz59o4 |
| `cheng2022mask2former` | Verified | CVF confirms CVPR 2022, pp. 1290-1299: https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html |
| `zhou2017ade20k` | Verified | CVF confirms CVPR 2017, pp. 633-641: https://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_Scene_Parsing_Through_CVPR_2017_paper.html |
| `cordts2016cityscapes` | Verified | CVF confirms CVPR 2016, pp. 3213-3223: https://openaccess.thecvf.com/content_cvpr_2016/html/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.html |

## Optional Updates

- `pix2next2024` is a valid 2024 arXiv citation. A peer-reviewed journal version
  was published in 2025 with DOI `10.3390/technologies13040154`. Update only if
  the report should cite the journal version instead of the preprint.
- Several valid conference entries omit page ranges and DOIs. Adding them would
  improve completeness but is not required to establish that the references are
  real.
- Internal BibTeX keys such as `dong2015image`, `leonard2013survey`, and
  `saharia2022image` contain a different year from the final publication year.
  The actual `year` fields are correct, so these key names do not affect the
  rendered bibliography.

## Present but Not Cited

These seven verified entries are in `references.bib` but are not currently cited
by any `\cite{...}` command in `main.tex` or `chapters/*.tex`:

- `wang2023internimage`
- `zhang2000flexible`
- `hartley2004multiple`
- `howard2019mobilenetv3`
- `li2023bbdm`
- `zhang2018lpips`
- `lugaresi2019mediapipe`

The Biber log reports 42 citekeys and no bibliography-data warnings.
