from django.shortcuts import render
import os
import ast
from dotenv import load_dotenv

load_dotenv()


def index(request):
    options = ast.literal_eval(os.getenv("OPTIONS", "[]"))
    return render(request, 'index.html', {'options': options})

