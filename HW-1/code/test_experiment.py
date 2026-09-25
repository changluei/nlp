"""Small correctness checks for splitting, pooling, metrics, and train-only vocabulary."""
import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from experiment import tokenize, mean_vectors, prepare, metrics


class CorrectnessTests(unittest.TestCase):
    def test_tokenization(self):
        self.assertEqual(tokenize("Apple APPLE, don't 123."), ['apple','apple',"don't"])

    def test_pooling_counts_repeated_words_and_skips_oov(self):
        matrix, stats = mean_vectors([['a','a','b','oov'],['oov'],[]],
            {'a':np.array([1.,0.]),'b':np.array([0.,3.])},dimension=2)
        np.testing.assert_allclose(matrix,[[2/3,1.],[0,0],[0,0]])
        self.assertEqual(stats['empty_documents'],2)
        self.assertAlmostEqual(stats['token_coverage'],3/5)

    def test_vocabulary_has_no_validation_terms(self):
        v = CountVectorizer(tokenizer=tokenize,token_pattern=None)
        v.fit(['train train shared'])
        self.assertEqual(v.transform(['heldout shared']).nnz,1)
        self.assertNotIn('heldout',v.vocabulary_)

    def test_split_disjoint_complete_reproducible_and_guarded(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with (root/'nyt.csv').open('w') as f:
                w=csv.writer(f);w.writerow(['text','label'])
                w.writerows((f'text item {i}', ['business','politics','sports'][i%3]) for i in range(30))
            a=prepare(root,root/'results',42)[2]
            b=prepare(root,root/'results',42)[2]
            self.assertEqual([len(a[k]) for k in a],[24,3,3])
            self.assertEqual(len(set(np.concatenate(list(a.values())))),30)
            for k in a: np.testing.assert_array_equal(a[k],b[k])
            with self.assertRaises(ValueError): prepare(root,root/'results',43)

    def test_macro_f1_includes_absent_predictions(self):
        result=metrics(np.array([0,1,2,2]),np.array([2,2,2,2]))
        self.assertEqual(result['accuracy'],.5)
        self.assertAlmostEqual(result['macro_f1'],2/9)


if __name__=='__main__': unittest.main()
