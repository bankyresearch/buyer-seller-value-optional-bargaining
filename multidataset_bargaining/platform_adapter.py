"""Non-production, single-item quote-state demonstration; no payment or UI API.

Integer cents only. The seller sees its floor, listing, round and customer bid,
not customer value or deadline. Production requires persistent, atomic storage.
"""
from dataclasses import dataclass,replace
from math import ceil

@dataclass(frozen=True)
class Quote:
    token: str
    product: str
    catalog_version: int
    cents: int
    expires_at: int
    consumed: bool=False

def numeric_offer(text):
    import re
    if not re.fullmatch(r'\d+(?:\.\d{1,2})?',text.strip()):
        raise ValueError('Use a single nonnegative amount with at most two decimals')
    from decimal import Decimal
    return int(Decimal(text.strip())*100)

def quote_price(listing,floor,previous,round_number,horizon,buyer_bid):
    values=[listing,floor,previous,round_number,horizon,buyer_bid]
    if any(type(v) is not int for v in values):raise ValueError('Integer inputs required')
    if not (0<=floor<=previous<=listing and listing>0 and 1<=round_number<=horizon and buyer_bid>=0):raise ValueError('Invalid bargaining state')
    time_ask=ceil(listing-(listing-floor)*round_number/horizon)
    ask=max(floor,min(previous,time_ask))
    # Accept an incoming bid meeting the ask, subject to the posted-price ceiling.
    return min(buyer_bid,listing) if buyer_bid>=ask else ask

def confirm(quote,*,now,product,catalog_version,current_floor,stock_available,explicit_confirmation):
    if not explicit_confirmation:raise ValueError('Customer confirmation required')
    if quote.consumed:raise ValueError('Quote already consumed')
    if now>=quote.expires_at:raise ValueError('Quote expired')
    if quote.product!=product or quote.catalog_version!=catalog_version:raise ValueError('Catalogue changed')
    if not stock_available or quote.cents<current_floor:raise ValueError('Merchant conditions changed')
    return replace(quote,consumed=True)

if __name__=='__main__':
    amount=quote_price(10000,7000,10000,1,8,numeric_offer('75.00'))
    q=Quote('illustration-only','SKU-1',1,amount,300)
    print(confirm(q,now=10,product='SKU-1',catalog_version=1,current_floor=7000,stock_available=True,explicit_confirmation=True))
