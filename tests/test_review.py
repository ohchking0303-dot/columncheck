import contextlib,io,json,tempfile,unittest
from pathlib import Path
from csv_review import train_model,suggest,valid,review,render_html,baseline,main,normalize,ROOT

class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.model=train_model()
    def run_csv(self,text):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'input.csv';p.write_text(text,encoding='utf-8');before=p.read_bytes()
            result=review(p,self.model)
            self.assertEqual(before,p.read_bytes())
            return result
    def test_learned_classifier_runs(self):
        self.assertEqual(suggest(self.model,'customer_email')['suggestion'],'email')
        self.assertGreater(len(self.model[-1].coef_[0]),100)
    def test_unknown_abstains(self): self.assertEqual(suggest(self.model,'漢字')['suggestion'],'review_required')
    def test_normalization(self): self.assertEqual(normalize('invoiceDueDate'),'invoice due date')
    def test_no_train_test_header_overlap(self):
        sets=[{normalize(x['header']) for x in json.loads((ROOT/'data'/f'{s}.json').read_text())} for s in ['train','holdout']]
        self.assertFalse(sets[0]&sets[1])
    def test_email(self):
        self.assertTrue(valid('a@example.test','email'));self.assertFalse(valid('bad','email'))
    def test_date(self):
        self.assertTrue(valid('2024-02-29','date'));self.assertFalse(valid('2026-02-30','date'));self.assertFalse(valid('02/03/26','date'))
    def test_money_finite(self):
        for x in ['NaN','Infinity','1,000','$10','1e3']: self.assertFalse(valid(x,'money'))
        self.assertTrue(valid('-10.50','money'))
    def test_quantity_contract(self):
        self.assertTrue(valid('0','quantity'));self.assertFalse(valid('-1','quantity'));self.assertFalse(valid('2.5','quantity'))
    def test_identifier_unvalidated(self): self.assertIsNone(valid('00012','identifier'))
    def test_empty(self):
        with self.assertRaisesRegex(ValueError,'empty'): self.run_csv('')
    def test_duplicate_headers(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'): self.run_csv('id,id\n1,2\n')
    def test_empty_header(self):
        with self.assertRaisesRegex(ValueError,'nonempty'): self.run_csv('id,\n1,2\n')
    def test_ragged(self):
        with self.assertRaisesRegex(ValueError,'column count'): self.run_csv('id,email\n1\n')
    def test_bom_and_quoted_commas(self): self.assertEqual(self.run_csv('\ufeffnotes\n"a,b"\n')['rows'],1)
    def test_missing_and_invalid_separate(self):
        c=self.run_csv('email\n""\nbad\na@example.test\n')['columns'][0]
        self.assertEqual(c['missing_record_numbers'],[2]);self.assertEqual(c['invalid_record_numbers'],[3]);self.assertEqual(c['checked_values'],2)
    def test_csv_injection_never_executed(self):
        report=self.run_csv('notes\n=1+1\n');self.assertNotIn('=1+1',json.dumps(report))
    def test_html_escape(self):
        out=render_html({'header':'<script>alert(1)</script>'});self.assertNotIn('<script>',out);self.assertIn('&lt;script&gt;',out)
    def test_source_hash(self):
        import hashlib
        text='id\n001\n';self.assertEqual(self.run_csv(text)['source_sha256'],hashlib.sha256(text.encode()).hexdigest())
    def test_baseline_ambiguous(self): self.assertEqual(baseline('item total count'),'review_required')
    def test_limit_rows(self):
        with self.assertRaisesRegex(ValueError,'10000'): self.run_csv('notes\n'+'x\n'*10001)
    def test_limit_columns(self):
        with self.assertRaisesRegex(ValueError,'100 columns'): self.run_csv(','.join('x'+str(i) for i in range(101)))
    def test_empty_data_has_no_validation(self):
        c=self.run_csv('email\n')['columns'][0];self.assertEqual(c['validator'],'no_format_validation')
    def test_cli_refuses_input_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'a.csv';p.write_text('id\n1\n')
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): main([str(p),'--json-out',str(p)])
            self.assertEqual(p.read_text(),'id\n1\n')
    def test_cli_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'a.csv';p.write_text('id\n1\n');q=Path(folder)/'out.json';q.write_text('keep')
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): main([str(p),'--json-out',str(q)])
            self.assertEqual(q.read_text(),'keep')
    def test_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'a.csv';p.write_text('email\nbad\n');q=Path(folder)/'out.json'
            with contextlib.redirect_stdout(io.StringIO()): main([str(p),'--json-out',str(q)])
            self.assertEqual(json.loads(q.read_text())['columns'][0]['invalid_record_numbers'],[2])

if __name__=='__main__': unittest.main()
