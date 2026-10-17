INSERT INTO erp.vendors(vendor_id,vendor_name,country,currency) VALUES
('VEND-001','Global Logistics Ltd','UK','USD'),
('VEND-002','BlueOcean Transport Co.','USA','USD'),
('VEND-003','Transporte Ibérico S.A.','Spain','EUR'),
('VEND-004','HafenLogistik GmbH','Germany','EUR'),
('VEND-005','SwiftMove Couriers','USA','USD'),
('VEND-006','OceanFreight Pvt','India','INR')
ON CONFLICT (vendor_id) DO NOTHING;

INSERT INTO erp.sku_master(item_code,category,uom,gst_rate) VALUES
('SKU-001','Packaging','roll',10.0),
('SKU-002','Safety','pair',10.0),
('SKU-003','Safety','piece',10.0),
('SKU-101','Logistics','piece',10.0),
('SKU-201','Packaging','box',10.0),
('SKU-202','Packaging','label',10.0),
('SKU-301','Transport','crate',10.0),
('SKU-302','Safety','vest',10.0),
('SKU-401','Charges','service',10.0),
('SKU-402','Packaging','box',10.0),
('SKU-501','Transport','Service',10.0)
ON CONFLICT (item_code) DO NOTHING;

INSERT INTO erp.purchase_orders(po_number,vendor_id) VALUES
('PO-1001','VEND-001'),
('PO-1002','VEND-002'),
('PO-1003','VEND-003'),
('PO-1004','VEND-004'),
('PO-1005','VEND-005'),
('PO-1006','VEND-006')
ON CONFLICT (po_number) DO NOTHING;

INSERT INTO erp.po_line_items(po_number,item_code,description,quantity,unit_price,currency) VALUES
('PO-1001','SKU-001','Pallet Wrapping Film', 50.000,12.0000,'USD'),
('PO-1001','SKU-002','Industrial Gloves', 120.000,3.0000,'USD'),
('PO-1001','SKU-003','Safety Helmets', 30.000,15.0000,'USD'),

('PO-1002','SKU-101','Container Seals', 200.000,1.2500,'USD'),

('PO-1003','SKU-201','Cajas de carton', 100.000,2.5000,'EUR'),
('PO-1003','SKU-202','Etiquetas adhesivas', 300.000,0.3000,'EUR'),

('PO-1004','SKU-301','Transportkisten', 40.000,25.0000,'EUR'),
('PO-1004','SKU-302','Sicherheitswesten', 60.000,10.0000,'EUR'),

('PO-1005','SKU-401','Fuel Surcharge', 1.000,120.0000,'USD'),
('PO-1005','SKU-402','Express Delivery Boxes', 25.000,4.5000,'USD'),

('PO-1006','SKU-501','Container Transport', 5.000,8000.0000,'INR');



