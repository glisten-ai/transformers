# Doordash Experiments

## Cuisine-type

`experiments/text-classification/run_doordash.py` is used to run a classification doordash model

`notebooks/Evaluate with model` has evaluation of the above

With argmax over all labels, 50% accuracy

With argmax over target labels, 70% accuracy (below)

```
** BBQ Pork Bao true: chinese pred: japanese
Steak & Cheese true: american pred: italian
Mission true: mexican pred: japanese
** Spicy Noodles true: thai pred: japanese
Tikka Masala Burrito true: indian pred: mexican
** Samosas true: indian pred: japanese
** Jumbo Wings (10-pc Regular Pack) true: american pred: italian
** Tom Yum Goong (Spicy Shrimp Lemongrass Soup) true: thai pred: japanese
** Pad Woon Sen true: thai pred: japanese
** Malabar Parota true: indian pred: other
The Meats true: italian pred: other
** Vegetable Briyani true: indian pred: japanese
Turkey Breast Footlong Regular Sub true: other pred: american
** Buffalo Chicken Sandwich true: american pred: other
** Tingly Mapo Tofu true: chinese pred: japanese
** Wonton Soup true: chinese pred: japanese
Nashville Looks So Good on You true: american pred: other
** Kimchee Stew true: other pred: japanese
** Vegan Caesar true: other pred: mexican
McCovey true: american pred: other
** Gus's Special true: american pred: mexican
** Daily Lentil Soup true: other pred: american
** Tom Yum Noodle Soup true: thai pred: japanese
Pepperoni true: american pred: italian
Southwest BBQ Chicken true: other pred: american
** 10. Hawaiian true: italian pred: japanese
=====
0.36 (percentage wrong)
wrong/total = 26/72 = 0.36
**/total = 17/72 = 0.236
```
