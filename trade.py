import requests as req
import pandas as pd
import cbpro
import math as mt
from datetime import timedelta

def order(pos):
    pc = cbpro.PublicClient()
