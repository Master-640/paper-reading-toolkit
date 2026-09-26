# UniCM 论文 ——《Learning the coupled dynamics of global climate modes》


> 来源：Nature 论文页**可公开获取**部分（正文全文需订阅）+ 补充材料。抓取时间：2026-09-25

> 页面：https://www.nature.com/articles/s42256-026-01245-5


## 0. 书目信息


- **标题**：Learning the coupled dynamics of global climate modes

- **作者**：Yuan Yuan; Jingtao Ding; Zhongpu Qiu; Jingfang Fan; Yong Li

- **单位**：
  1. Department of Electronic Engineering, Tsinghua University, Beijing, China
  2. Beijing National Research Center for Information Science and Technology, Tsinghua University, Beijing, China
  3. School of Systems Science, Beijing Normal University, Beijing, China
  4. Potsdam Institute for Climate Impact Research (PIK), Potsdam, Germany

- **期刊**：Nature Machine Intelligence, Vol. 8, pp. 930–941 (2026)

- **在线日期**：2026-06-01

- **DOI**：10.1038/s42256-026-01245-5

- **代码**：https://github.com/tsinghua-fib-lab/UniCM-Global-Climate-Modes ；Zenodo https://doi.org/10.5281/zenodo.19173780


---

## 1. 摘要（原文 Abstract，逐字）


Global weather extremes, from monsoons to droughts, are shaped by a network of recurrent, coupled ocean–atmosphere patterns known as climate modes. These modes, spanning from the Pacific’s El Niño-Southern Oscillation to interconnected patterns in the Indian and Atlantic Oceans, form a dynamically linked global system governed by complex nonlinear interactions. Holistically forecasting this interconnected system—rather than treating modes in isolation or in simplified pairs as existing approaches do—remains a fundamental challenge in machine intelligence for complex systems. Here we introduce UniCM, a unified deep model for global climate modes forecasting. Its key innovation lies in a dual-branch architecture that learns the dynamics of a coupled system directly from data, achieving a truly unified prediction through the synergistic modelling of localized dynamics and their collective global couplings. UniCM achieves strong performance in unified global climate-mode forecasting, outperforming strong existing baselines and extending the skilful forecast lead time across multiple major climate modes. It successfully captured the diversity of historical events, from the extreme 1997–1998 El Niño to the prolonged and challenging 2020–2023 triple-dip La Niña. Beyond accuracy, UniCM offers interpretability; its internal attention mechanism identifies dynamic precursors and quantifies the structured inter-mode interactions that precede extreme climate events. Our results demonstrate that learning the coupled dynamics of climate modes as an interconnected system unlocks emergent predictability, laying a foundation for unified forecasting and data-driven insights that deepen our understanding of global ocean–atmosphere dynamics.


---

## 2. 论文页上可获取的章节骨架


- Abstract
- Inline Recommendations
- Data availability
- Code availability
- References
- Acknowledgements
- Author information
- Ethics declarations
- Peer review
- Additional information
- Supplementary information
- Rights and permissions
- About this article


---

## 3. 主图标题（Figure captions；高清图见 UniCM_Fig1-5.png）


- **Fig1**：Fig. 1: Overview of the UniCM architecture.
- **Fig2**：Fig. 2: ENSO forecast performance of UniCM.
- **Fig3**：Fig. 3: UniCM’s performance on forecasting global climate modes.
- **Fig4**：Fig. 4: Spatial prediction skill of SST across different lead times.
- **Fig5**：Fig. 5: UniCM’s attention mechanism reveals event-specific precursors to major ENSO events.


---

## 4. 补充材料清单（官方说明）


- **MOESM1**：Supplementary Notes (Sections 1–6), Tables 1–4, Figs. 1–41 and References.
  下载：https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-026-01245-5/MediaObjects/42256_2026_1245_MOESM1_ESM.pdf
- **MOESM2**：
  下载：https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-026-01245-5/MediaObjects/42256_2026_1245_MOESM2_ESM.pdf


---

## 5. 数据 / 代码可用性与声明（原文）


**Data availability**：All datasets used in this study are publicly available. The CMIP6 data are available from https://cds.climate.copernicus.eu/datasets/projections-cmip6; the ORAS5 data are available from the Climate Data Store (CDS) at https://cds.climate.copernicus.eu/datasets/reanalysis-oras5; the ERA5 data are available from CDS at https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means; the GODAS data are available from the National Oceanic Atmospheric Administration Physical Science Laboratory at https://psl.noaa.gov/data/gridded/data.godas.html; and the SODA data are available from the SODA project at https://soda.umd.edu/.


**Code availability**：The source code for UniCM is publicly available via GitHub at https://github.com/tsinghua-fib-lab/UniCM-Global-Climate-Modes and via Zenodo at https://doi.org/10.5281/zenodo.19173780 (ref. 67).


**Acknowledgements（基金）**：This work was supported in part by the National Key Research and Development Program of China (grant no. 2024YFC3307603), the National Natural Science Foundation of China (grant no. 62476152) and the Tsinghua–Toyota Joint Research Institute Interdisciplinary Program to Y.L. and J.D. J.F. received support from the National Natural Science Foundation of China (grant nos. T2525011, 42450183, 12275020, 12135003 and 42461144209) and the National Key R&D Program of China (grant no. 2025YFF0517203). J.F. acknowledges support from the Fundamental Research Funds for the Central Universities.


**Ethics declarations**：Competing interests The authors declare no competing interests.


**Peer review**：Peer review information Nature Machine Intelligence thanks Annalisa Bracco, Maximilian Gelbrecht and Jing-Jia Luo for their contribution to the peer review of this work. Peer reviewer reports are available.


---

## 6. 参考文献全表（共 67 条，原文格式）


1. Lobell, D. B., Schlenker, W. & Costa-Roberts, J. Climate trends and global crop production since 1980. Science 333, 616–620 (2011).
2. Dettinger, M. D. & Diaz, H. F. Global characteristics of stream flow seasonality and variability. J. Hydrometeorol. 1, 289–310 (2000).
3. Bracco, A. et al. Machine learning for the physics of climate. Nat. Rev. Phys. 7, 6–20 (2025).
4. Philander, S. G. H. El Nino Southern Oscillation phenomena. Nature 302, 295–301 (1983).
5. IPCC Climate Change 2021: The Physical Science Basis (eds Masson-Delmotte, V. et al.) (Cambridge Univ. Press, 2021).
6. Deser, C., Alexander, M. A., Xie, S.-P. & Phillips, A. S. Sea surface temperature variability: patterns and mechanisms. Annu. Rev. Mar. Sci. 2, 115–143 (2010).
7. McPhaden, M. J., Zebiak, S. E. & Glantz, M. H. ENSO as an integrating concept in earth science. Science 314, 1740–1745 (2006).
8. Timmermann, A. et al. El Niño–Southern Oscillation complexity. Nature 559, 535–545 (2018).
9. Cai, W. et al. Changing El Niño–Southern Oscillation in a warming climate. Nat. Rev. Earth Environ. 2, 628–644 (2021).
10. Zhou, L. & Zhang, R.-H. A self-attention–based neural network for three-dimensional multivariate modeling and its skillful ENSO predictions. Sci. Adv. 9, eadf2827 (2023).
11. Saji, N., Goswami, B. N., Vinayachandran, P. & Yamagata, T. A dipole mode in the tropical Indian Ocean. Nature 401, 360–363 (1999).
12. Webster, P. J., Moore, A. M., Loschnigg, J. P. & Leben, R. R. Coupled ocean–atmosphere dynamics in the Indian Ocean during 1997–98. Nature 401, 356–360 (1999).
13. Zhao, S. et al. Explainable El Niño predictability from climate mode interactions. Nature 630, 891–898 (2024).
14. Shaw, T. A. & Stevens, B. The other climate crisis. Nature 639, 877–887 (2025).
15. Cai, W. et al. Pantropical climate interactions. Science 363, eaav4236 (2019).
16. Jin, Y. et al. The Indian Ocean weakens the ENSO spring predictability barrier: role of the Indian Ocean basin and dipole modes. J. Clim. 36, 8331–8345 (2023).
17. Jo, H.-S. et al. Southern Indian Ocean dipole as a trigger for Central Pacific El Niño since the 2000s. Nat. Commun. 13, 6965 (2022).
18. Alexander, M. A., Shin, S.-I. & Battisti, D. S. The influence of the trend, basin interactions, and ocean dynamics on tropical ocean prediction. Geophys. Res. Lett. 49, e2021GL096120 (2022).
19. Cai, W., Van Rensch, P., Cowan, T. & Hendon, H. H. Teleconnection pathways of ENSO and the IOD and the mechanisms for impacts on Australian rainfall. J. Clim. 24, 3910–3923 (2011).
20. Dijkstra, H. A. Nonlinear Climate Dynamics (Cambridge Univ. Press, 2013).
21. Chiang, J. C. & Vimont, D. J. Analogous Pacific and Atlantic meridional modes of tropical atmosphere–ocean variability. J. Clim. 17, 4143–4158 (2004).
22. Vos, E., Huybers, P. & Tziperman, E. Climate change alters teleconnections. Geophys. Res. Lett. 53, e2025GL119307 (2026).
23. Ling, F. et al. Multi-task machine learning improves multi-seasonal prediction of the Indian Ocean dipole. Nat. Commun. 13, 7681 (2022).
24. Luo, J.-J. et al. Interaction between El Nino and extreme Indian Ocean dipole. J. Clim. 23, 726–742 (2010).
25. Zhang, Z. et al. Enhancing the predictability limits of ENSO with physics-guided deep echo state networks. NPJ Clim. Atmos. Sci. 9, 92 (2026).
26. Wang, B. et al. Understanding the recent increase in multiyear La Niñas. Nat. Clim. Change 13, 1075–1081 (2023).
27. Heede, U. K. & Fedorov, A. V. Colder eastern Equatorial Pacific and stronger Walker circulation in the early 21st century: separating the forced response to global warming from natural variability. Geophys. Res. Lett. 50, e2022GL101020 (2023).
28. Jiang, S., Zhu, C., Hu, Z.-Z., Jiang, N. & Zheng, F. Triple-dip La Niña in 2020–23: understanding the role of the annual cycle in tropical Pacific SST. Environ. Res. Lett. 18, 084002 (2023).
29. Liang, Y., Xie, S.-P., Fedorov, A. & Yeager, S. G. North pacific meridional mode has larger impacts on El Niño evolution than the March Madden–Julian oscillation. Sci. Adv. 11, eadv8621 (2025).
30. Bassett, D. S. & Sporns, O. Network neuroscience. Nat. Neurosci. 20, 353–364 (2017).
31. Helbing, D. Globally networked risks and how to respond. Nature 497, 51–59 (2013).
32. Allesina, S. & Tang, S. Stability criteria for complex ecosystems. Nature 483, 205–208 (2012).
33. Reichstein, M. et al. Deep learning and process understanding for data-driven earth system science. Nature 566, 195–204 (2019).
34. Ham, Y.-G., Kim, J.-H. & Luo, J.-J. Deep learning for multi-year ENSO forecasts. Nature 573, 568–572 (2019).
35. Lyu, P. et al. Resonet: robust and explainable ENSO forecasts with hybrid convolution and transformer networks. Adv. Atmos. Sci. 41, 1289–1298 (2024).
36. Duan, W., Liu, X., Zhu, K. & Mu, M. Exploring the initial errors that cause a significant ‘spring predictability barrier’ for El Niño events. J. Geophys. Res. Oceans 114, C04022 (2009).
37. Jin, F.-F. An equatorial ocean recharge paradigm for ENSO. Part I: conceptual model. J. Atmos. Sci. 54, 811–829 (1997).
38. Ashok, K., Behera, S. K., Rao, S. A., Weng, H. & Yamagata, T. El Niño Modoki and its possible teleconnection. J. Geophys. Res. Oceans 112, (2007).
39. Kao, H.-Y. & Yu, J.-Y. Contrasting Eastern-Pacific and Central-Pacific types of ENSO. J. Clim. 22, 615–632 (2009).
40. Simpson, I. R. et al. Confronting Earth system model trends with observations. Sci. Adv. 11, eadt8035 (2025).
41. Ham, Y.-G., Kug, J.-S., Park, J.-Y. & Jin, F.-F. Sea surface temperature in the north tropical Atlantic as a trigger for El Niño/Southern Oscillation events. Nat. Geosci. 6, 112–116 (2013).
42. Guo, F., Liu, Q., Yang, J. & Fan, L. Three types of Indian Ocean basin modes. Clim. Dyn. 51, 4357–4370 (2018).
43. Zhang, H., Clement, A. & Di Nezio, P. The South Pacific meridional mode: a mechanism for ENSO-like variability. J. Clim. 27, 769–783 (2014).
44. Chen, L. et al. A machine learning model that outperforms conventional global subseasonal forecast models. Nat. Commun. 15, 6425 (2024).
45. Wang, L. et al. Cas-Canglong: a skillful 3D transformer model for sub-seasonal to seasonal global sea surface temperature prediction. Preprint at https://doi.org/10.48550/arXiv.2409.05369 (2024).
46. Alexander, M. A. et al. The atmospheric bridge: the influence of ENSO teleconnections on air–sea interaction over the global oceans. J. Clim. 15, 2205–2231 (2002).
47. Schwing, F. B., Murphree, T., deWitt, L. & Green, P. M. The evolution of oceanic and atmospheric anomalies in the Northeast Pacific during the El Niño and La Niña events of 1995–2001. Prog. Oceanogr. 54, 459–491 (2002).
48. Fan, J., Meng, J., Ashkenazy, Y., Havlin, S. & Schellnhuber, H. J. Network analysis reveals strongly localized impacts of El Niño. Proc. Natl Acad. Sci. USA 114, 7543–7548 (2017).
49. Meng, J. et al. Complexity-based approach for El Niño magnitude forecasting before the spring predictability barrier. Proc. Natl Acad. Sci. USA 117, 177–183 (2020).
50. Chen, Y. et al. Combined dynamical-deep learning ENSO forecasts. Nat. Commun. 16, 3845 (2025).
51. Zscheischler, J. et al. Future climate risk from compound events. Nat. Clim. Change 8, 469–477 (2018).
52. AghaKouchak, A. et al. Climate extremes and compound hazards in a warming world. Annu. Rev. Earth Planet. Sci. 48, 519–548 (2020).
53. Raymond, C. et al. Understanding and managing connected extreme events. Nat. Clim. Change 10, 611–621 (2020).
54. McKenna, S., Santoso, A., Gupta, A. S., Taschetto, A. S. & Cai, W. Indian Ocean dipole in CMIP5 and CMIP6: characteristics, biases, and links to ENSO. Sci. Rep. 10, 11500 (2020).
55. Yang, Y. et al. Seasonality and predictability of the Indian Ocean dipole mode: ENSO forcing and internal variability. J. Clim. 28, 8021–8036 (2015).
56. Zhang, C. Madden–Julian oscillation. Rev. Geophys. 43, (2005).
57. Rahmstorf, S. Ocean circulation and climate during the past 120,000 years. Nature 419, 207–214 (2002).
58. Buckley, M. W. & Marshall, J. Observations, inferences, and mechanisms of the Atlantic meridional overturning circulation: a review. Rev. Geophys. 54, 5–63 (2016).
59. Schmidt, M. & Lipson, H. Distilling free-form natural laws from experimental data. Science 324, 81–85 (2009).
60. Udrescu, S.-M. & Tegmark, M. AI Feynman: a physics-inspired method for symbolic regression. Sci. Adv. 6, eaay2631 (2020).
61. Camps-Valls, G. et al. Discovering causal relations and equations from data. Phys. Rep. 1044, 1–68 (2023).
62. Zanna, L. & Bolton, T. Data-driven equation discovery of ocean mesoscale closures. Geophys. Res. Lett. 47, e2020GL088376 (2020).
63. Chinazzi, M. et al. The effect of travel restrictions on the spread of the 2019 novel coronavirus (COVID-19) outbreak. Science 368, 395–400 (2020).
64. Rohr, R. P., Saavedra, S. & Bascompte, J. On the structural stability of mutualistic systems. Science 345, 1253497 (2014).
65. Hirahara, S., Balmaseda, M. A., Boisseson, E. & Hersbach, H. Sea Surface Temperature and Sea Ice Concentration for ERA5 Vol. 26 (European Centre for Medium Range Weather Forecasts, 2016).
66. Zuo, H., Balmaseda, M. A., Tietsche, S., Mogensen, K. & Mayer, M. The ECMWF operational ensemble reanalysis–analysis system for ocean and sea ice: a description of the system and assessment. Ocean Sci. 15, 779–808 (2019).
67. Yuan, Y., Ding, J., Qiu, Z., Fan, J. & Li, Y. Code for learning the coupled dynamics of global climate modes. Zenodo https://doi.org/10.5281/zenodo.19173780 (2026).