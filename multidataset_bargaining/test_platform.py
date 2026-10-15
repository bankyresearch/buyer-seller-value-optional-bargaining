import unittest
from platform_adapter import numeric_offer,quote_price,Quote,confirm
class AdapterTests(unittest.TestCase):
    def test_numeric(self):self.assertEqual(numeric_offer('75.01'),7501)
    def test_ambiguous_input(self):
        for x in ['75 or 80','-1','1.001','NaN','give me 80']:
            with self.assertRaises(ValueError):numeric_offer(x)
    def test_bounds(self):
        for t in range(1,9):
            for bid in range(0,13000,100):self.assertTrue(7000<=quote_price(10000,7000,10000,t,8,bid)<=10000)
    def test_floor(self):self.assertEqual(quote_price(10000,7000,9000,8,8,6000),7000)
    def test_bad_state(self):
        with self.assertRaises(ValueError):quote_price(10000,11000,10000,1,8,5000)
    def setUp(self):
        self.q=Quote('token','SKU',1,8000,300)
        self.kw=dict(now=10,product='SKU',catalog_version=1,current_floor=7000,stock_available=True,explicit_confirmation=True)
    def test_confirm(self):self.assertTrue(confirm(self.q,**self.kw).consumed)
    def test_expired(self):
        with self.assertRaises(ValueError):confirm(self.q,**{**self.kw,'now':300})
    def test_replay(self):
        with self.assertRaises(ValueError):confirm(confirm(self.q,**self.kw),**self.kw)
    def test_version(self):
        with self.assertRaises(ValueError):confirm(self.q,**{**self.kw,'catalog_version':2})
    def test_stock(self):
        with self.assertRaises(ValueError):confirm(self.q,**{**self.kw,'stock_available':False})
    def test_floor_changed(self):
        with self.assertRaises(ValueError):confirm(self.q,**{**self.kw,'current_floor':9000})
    def test_confirmation(self):
        with self.assertRaises(ValueError):confirm(self.q,**{**self.kw,'explicit_confirmation':False})
if __name__=='__main__':unittest.main()
