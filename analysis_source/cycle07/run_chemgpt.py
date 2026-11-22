import sys
import shutil
from pathlib import Path

root=Path(__file__).resolve().parent
source=root.parent/'cycle06'
sys.path[:0]=[str(root.parent/'cycle07_model'),str(source)]
from frozen_chemgpt import FrozenChemGPTEncoder
import fit
import fit_continuous
import evaluate
import evaluate_continuous

output=root/'chemgpt'
output.mkdir(exist_ok=True)
projects=['pxr','cyp2c9','cyp1a2','cyp2d6','cyp3a4']
encoder=FrozenChemGPTEncoder()
for module in (fit,fit_continuous,evaluate,evaluate_continuous):
    module.root=output
for project in projects:
    target=output/project
    target.mkdir(exist_ok=True)
    for file in ('structures.csv','early_labels.csv','late_labels.csv'):
        shutil.copyfile(source/project/file,target/file)
    fit.fit(project,encoder)
    evaluate.evaluate(project)
    fit_continuous.fit_project(project,encoder)
    evaluate_continuous.evaluate(project)
