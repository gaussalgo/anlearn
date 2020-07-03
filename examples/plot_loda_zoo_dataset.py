"""
Explaining the cause of an anomaly using LODA on Zoo dataset
============================================================
In this example we're using LODA [1]_ on Zoo dataset [3]_ to show how we could
explain the cause of an anomaly.

"""


# %%

# Author: Ondrej Kurák kurak@gaussalgo.com
# License: LGPLv3+

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP

from anlearn.loda import LODA

frame = pd.read_csv(
    "../datasets/zoo.data",
    names=[
        "animals",
        "hair",
        "feathers",
        "eggs",
        "milk",
        "airborne",
        "aquatic",
        "predator",
        "toothed",
        "backbone",
        "breathes",
        "venomous",
        "fins",
        "legs",
        "tail",
        "domestic",
        "catsize",
        "type",
    ],
)

frame.set_index("animals", inplace=True)

print(frame)


# %%


# !cat ../datasets/zoo.names

# 1. Title: Zoo database

# 2. Source Information
#    -- Creator: Richard Forsyth
#    -- Donor: Richard S. Forsyth
#              8 Grosvenor Avenue
#              Mapperley Park
#              Nottingham NG3 5DX
#              0602-621676
#    -- Date: 5/15/1990

# 3. Past Usage:
#    -- None known other than what is shown in Forsyth's PC/BEAGLE User's Guide.

# 4. Relevant Information:
#    -- A simple database containing 17 Boolean-valued attributes.  The "type"
#       attribute appears to be the class attribute.  Here is a breakdown of
#       which animals are in which type: (I find it unusual that there are
#       2 instances of "frog" and one of "girl"!)

#       Class# Set of animals:
#       ====== ===============================================================
#            1 (41) aardvark, antelope, bear, boar, buffalo, calf,
#                   cavy, cheetah, deer, dolphin, elephant,
#                   fruitbat, giraffe, girl, goat, gorilla, hamster,
#                   hare, leopard, lion, lynx, mink, mole, mongoose,
#                   opossum, oryx, platypus, polecat, pony,
#                   porpoise, puma, pussycat, raccoon, reindeer,
#                   seal, sealion, squirrel, vampire, vole, wallaby,wolf
#            2 (20) chicken, crow, dove, duck, flamingo, gull, hawk,
#                   kiwi, lark, ostrich, parakeet, penguin, pheasant,
#                   rhea, skimmer, skua, sparrow, swan, vulture, wren
#            3 (5)  pitviper, seasnake, slowworm, tortoise, tuatara
#            4 (13) bass, carp, catfish, chub, dogfish, haddock,
#                   herring, pike, piranha, seahorse, sole, stingray, tuna
#            5 (4)  frog, frog, newt, toad
#            6 (8)  flea, gnat, honeybee, housefly, ladybird, moth, termite, wasp
#            7 (10) clam, crab, crayfish, lobster, octopus,
#                   scorpion, seawasp, slug, starfish, worm

# 5. Number of Instances: 101

# 6. Number of Attributes: 18 (animal name, 15 Boolean attributes, 2 numerics)

# 7. Attribute Information: (name of attribute and type of value domain)
#    1. animal name:      Unique for each instance
#    2. hair		Boolean
#    3. feathers		Boolean
#    4. eggs		Boolean
#    5. milk		Boolean
#    6. airborne		Boolean
#    7. aquatic		Boolean
#    8. predator		Boolean
#    9. toothed		Boolean
#   10. backbone		Boolean
#   11. breathes		Boolean
#   12. venomous		Boolean
#   13. fins		Boolean
#   14. legs		Numeric (set of values: {0,2,4,5,6,8})
#   15. tail		Boolean
#   16. domestic		Boolean
#   17. catsize		Boolean
#   18. type		Numeric (integer values in range [1,7])

# 8. Missing Attribute Values: None

# 9. Class Distribution: Given above


# %%
# Data visualization
# ------------------
# For data visualization we're using UMAP [2]_ to transform data to two dimensional space.

X = frame.values[:, :-1]

# Prepare data for visualization using UMAP
umap = UMAP(n_neighbors=15, min_dist=0.9, random_state=42)
transformed = umap.fit_transform(X)

plt.figure(figsize=(10, 10))
plt.subplot(111, aspect="auto")
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
)


for type in np.unique(frame["type"]):
    selected = transformed[frame["type"] == type]
    plt.scatter(selected[:, 0], selected[:, 1], label=type)

for name, x, y in zip(frame.index, transformed[:, 0], transformed[:, 1]):
    plt.annotate(name, (x, y), alpha=0.8, fontsize=10)

plt.title("Zoo dataset - animal types", fontsize=18)
plt.xticks(())
plt.yticks(())
plt.legend(title="Animal type", title_fontsize=15, fontsize=13)
plt.show()


# %%
# Explaining the cause of an anomaly
# ----------------------------------

loda = LODA(n_estimators=100, random_state=42)
loda.fit(X)

scores = loda.score_samples(X)
predicted = loda.predict(X)


plt.figure(figsize=(10, 10))
plt.subplot(111, aspect="auto")
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
)

X_n = transformed[predicted == 1]
X_a = transformed[predicted == -1]

plt.scatter(X_n[:, 0], X_n[:, 1], color="tab:orange", label="Inliners")
plt.scatter(X_a[:, 0], X_a[:, 1], color="tab:blue", label="Outliers")

for name, x, y in zip(frame.index[predicted == 1], X_n[:, 0], X_n[:, 1]):
    plt.annotate(name, (x, y), alpha=0.5, fontsize=12)

for name, x, y in zip(frame.index[predicted == -1], X_a[:, 0], X_a[:, 1]):
    plt.annotate(name, (x, y), fontsize=15, ha="right")

plt.title("Zoo dataset - anomalous examples", fontsize=18)
plt.legend(title="Predicted", title_fontsize=15, fontsize=13)
plt.xticks(())
plt.yticks(())
plt.show()

feature_scores = loda.score_features(X)

for animal, score, feature_score in zip(
    frame[predicted == -1].itertuples(),
    scores[predicted == -1],
    feature_scores[predicted == -1],
):
    name = animal[0]
    srt = np.argsort(feature_score)[::-1]

    print(f"{name} score: {score:.3f}")

    for feature, value, importance in zip(
        frame.columns[srt][:4], np.array(animal[1:])[srt], feature_score[srt]
    ):
        print(f"\t{feature} {value} ({importance:.2f})")


# %%
# References
# ----------
# .. [1] Pevný, T. Loda: Lightweight on-line detector of anomalies. Mach Learn 102, 275–304 (2016).
#         <https://doi.org/10.1007/s10994-015-5521-0>
# .. [2] McInnes, L., Healy, J., Saul, N., & Grossberger, L. (2018). UMAP: Uniform Manifold Approximation and Projection
#        The Journal of Open Source Software, 3(29), 861. <https://github.com/lmcinnes/umap/>
# .. [3] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
#        [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
#        <https://archive.ics.uci.edu/ml/datasets/Zoo>
