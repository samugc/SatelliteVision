import unittest
from classification.core import *
class CoreTests(unittest.TestCase):
    def rows(self):
        return [dict(image_id=str(i), filename=f'{i}.tif', label=CATEGORIES[i%2]) for i in range(20)]
    def test_split_partition_and_order(self):
        rows=self.rows(); a,b=historical_split(rows)
        self.assertEqual([x['image_id'] for x in b], ['0','17'])
        self.assertEqual(len(a),18)
        self.assertFalse({x['image_id'] for x in a}&{x['image_id'] for x in b})
    def test_predictions_join_by_id_not_order(self):
        rows=self.rows(); pred=[dict(image_id=x['image_id'],filename=x['filename'],predicted_label=x['label']) for x in rows]
        self.assertEqual(evaluate_records(rows,list(reversed(pred)))['accuracy'],1.)
        pred[0]['predicted_label']=CATEGORIES[1]
        out=evaluate_records(rows,pred)
        self.assertEqual(out['confusion_matrix'][0][1],1)
        self.assertEqual(len(out['errors']),1)
    def test_rejects_missing_duplicate_or_wrong_filename(self):
        rows=self.rows(); pred=[dict(image_id=x['image_id'],filename=x['filename'],predicted_label=x['label']) for x in rows]
        for bad in [pred[:-1],pred+[pred[0]],[dict(x,filename='bad') for x in pred]]:
            with self.assertRaises(ValueError):evaluate_records(rows,bad)
    def test_accumulation_remainder_and_scaling(self):
        self.assertEqual([i for i in range(5) if accumulation_boundary(i,5,2)],[1,3,4])
        self.assertEqual([accumulation_group_size(i,5,2) for i in range(5)],[2,2,2,2,1])
        self.assertEqual(sum(1/accumulation_group_size(i,5,2) for i in range(5)),3.)
    def test_annotation_join(self):
        doc={'images':{'a':dict(image_id='a',filename='a.tif'),'b':dict(image_id='b',filename='b.tif')},
             'annotations':{'0':dict(image_id='b',category_id=CATEGORIES[1]),'1':dict(image_id='a',category_id=CATEGORIES[0])}}
        self.assertEqual(records_from_annotations(doc)[0]['label'],CATEGORIES[0])
if __name__=='__main__':unittest.main()
