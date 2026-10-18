"""Create original synthetic fixtures; no external/customer dataset."""
import json
from pathlib import Path
ROOT=Path(__file__).parent
TRAIN={
'email':['email','email address','customer email','contact email','billing email','mail address','e-mail','email contact','recipient email','sender email','electronic mail','work email'],
'date':['date','invoice date','due date','created date','payment date','order date','delivery date','issued on','paid on','deadline','billing date','service date'],
'money':['amount','total amount','invoice total','price','unit price','subtotal','net amount','gross amount','balance due','fee','cost','payment amount'],
'identifier':['id','invoice id','customer id','order id','reference','reference number','sku','product code','account code','record identifier','transaction id','invoice number'],
'quantity':['quantity','qty','count','item count','units','units sold','number of items','stock quantity','hours worked','pieces','ordered quantity','headcount'],
'description':['description','notes','memo','comment','details','item description','service description','product name','customer name','summary','title','message']}
HOLDOUT={
'email':['client e-mail','contact mail address','recipient_email_address','supportEmail','inbox'],
'date':['invoice_due_date','date issued','fulfillmentDate','payment deadline','maturity'],
'money':['gross_total','amount payable','unit_cost','service fee','remittance'],
'identifier':['invoice_ref','supplierID','stock code','transaction reference','tracking key'],
'quantity':['ordered_qty','total units','number_of_pieces','items_count','volume'],
'description':['additional notes','product details','service memo','line_item_description','narrative']}
for name,data in [('train',TRAIN),('holdout',HOLDOUT)]:
 (ROOT/'data'/f'{name}.json').write_text(json.dumps([{'header':h,'label':label} for label,hs in data.items() for h in hs],indent=2)+'\n')
