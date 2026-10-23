Lire des papyrus carbonisés par le Vésuve en 79, sans jamais les dérouler : c'est l'objectif du Vesuvius Challenge.
J'y contribue avec Volumen, un moteur open source de « déroulement virtuel » écrit en Rust. 📜

Le contexte : les rouleaux d'Herculanum sont numérisés par tomographie synchrotron (ESRF), à raison de plusieurs téraoctets par rouleau. Pour lire l'encre, il faut échantillonner le volume 3D le long de la surface de chaque feuille de papyrus, puis l'aplatir en images 2D. Les outils actuels sont puissants, mais lourds à compiler et à déployer.

L'approche de Volumen :
• Lecture OME-Zarr « out-of-core » avec un cache LRU borné en mémoire, en local ou en streaming depuis le bucket public S3 du challenge
• Échantillonnage trilinéaire le long des normales, parallélisé avec Rayon, sur maillages Wavefront .obj et sur le format tifxyz de VC3D
• Un binaire compatible avec la syntaxe de vc_render, des bindings Python (PyO3) vers NumPy/PyTorch et une sortie TIFF ou OME-Zarr
• Un mode expérimental de déroulement analytique d'une spirale d'Archimède, avec champ de déformation optionnel

📊 Premier contrôle sur données réelles (Scroll 1, ESRF 7,91 µm) : sur un petit segment de référence (249 × 85 px), mon prototype de l'algorithme d'échantillonnage reproduit la couche officielle de Volume Cartographer avec une corrélation de Pearson de 0,995.

⚠️ En toute transparence : c'est un projet jeune. Ce contrôle a été fait avec le prototype Python, pas encore avec le binaire Rust, et sur un seul petit segment. Aucun gain de vitesse face aux outils existants n'a encore été mesuré proprement. Les tests à grande échelle restent à faire.

Membres de la communauté Vesuvius, développeurs Rust, spécialistes de l'imagerie : retours et contributions bienvenus 👇
https://github.com/silveremartin-dev/Volumen

#VesuviusChallenge #Rust #HumanitésNumériques #Archéologie #ImagerieScientifique #OpenSource

(Image suggérée : docs/assets/00_real_scroll1_comparison_panel.png)
