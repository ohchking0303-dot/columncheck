import unittest,tempfile,json,subprocess,sys
from pathlib import Path
from csv_review import review,train_model,ROOT

class SchemaTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls): cls.model=train_model()
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name); self.csv=self.root/'source.csv'
  self.csv.write_text('mystery,email\n2026-02-30,not-an-email\n2026-02-28,a@example.test\n')
 def test_override_and_model_preserved(self):
  before=self.csv.read_bytes(); r=review(self.csv,self.model,{'mystery':'date'})
  c=r['columns'][0]; self.assertEqual(c['effective_label'],'date'); self.assertEqual(c['label_source'],'user_schema')
  self.assertIn('suggestion',c); self.assertEqual(c['invalid_record_numbers'],[2]); self.assertEqual(before,self.csv.read_bytes())
 def test_disable_validation(self):
  c=review(self.csv,self.model,{'email':'review_required'})['columns'][1]
  self.assertEqual(c['checked_values'],0); self.assertEqual(c['invalid_record_numbers'],[])
 def test_unknown_column(self):
  with self.assertRaises(ValueError): review(self.csv,self.model,{'absent':'date'})
 def test_invalid_label(self):
  for label in ['python',5,[],{}]:
   with self.assertRaises(ValueError): review(self.csv,self.model,{'email':label})
 def test_schema_not_object(self):
  with self.assertRaises(ValueError): review(self.csv,self.model,[])
 def cli(self,text,*more):
  schema=self.root/'schema.json'; schema.write_text(text)
  return subprocess.run([sys.executable,str(ROOT/'csv_review.py'),str(self.csv),'--schema',str(schema),*more],capture_output=True,text=True)
 def test_cli_schema(self):
  r=self.cli('{"mystery":"date"}'); self.assertEqual(r.returncode,0); self.assertEqual(json.loads(r.stdout)['columns'][0]['label_source'],'user_schema')
 def test_cli_duplicate_key(self):
  r=self.cli('{"email":"email","email":"date"}'); self.assertEqual(r.returncode,2); self.assertIn('Duplicate',r.stderr)
 def test_cli_prevent_schema_overwrite(self):
  schema=self.root/'schema.json'; r=self.cli('{"email":"email"}','--json-out',str(schema)); self.assertEqual(r.returncode,2)
  self.assertEqual(schema.read_text(),'{"email":"email"}')
 def test_cli_malformed(self): self.assertEqual(self.cli('{oops').returncode,2)
 def test_cli_null(self): self.assertEqual(self.cli('null').returncode,2)
