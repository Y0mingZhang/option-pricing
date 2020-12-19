option-pricing
==============================
Python script for calculating European option pricing, and the Greeks.
## setup
Code is tested with Python>=3.8
## usage
```python
>>> from black_scholes import *
>>> strike_price, current_price = 52, 50
>>> interest = 0.12
>>> sigma = 0.3
>>> expiry_time = 0.25
>>> current_time = 0
>>> print(EuropeanCallBlackScholes(strike_price, current_price, interest, sigma, expiry_time, current_time))
5.057386759734403
>>> type = "call"
>>> option_type = "call"
>>> print(Theta(strike_price, current_price, interest, sigma, expiry_time, current_time, option_type))
-9.176605978901557

```
