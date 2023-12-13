# index_service/views.py
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PostDetailSerializer
import os
import ast
from dotenv import load_dotenv

load_dotenv()


class PostDetail(APIView):
    def get(self, request, option, format=None):
        options = ast.literal_eval(os.getenv("OPTIONS", "[]"))

        if option in options:
            serializer = PostDetailSerializer(data={'option': option})
            if serializer.is_valid():
                return Response(serializer.data)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response({'error': 'Option not found'}, status=status.HTTP_404_NOT_FOUND)


def index(request):
    options = ast.literal_eval(os.getenv("OPTIONS", "[]"))
    return render(request, 'index.html', {'options': options})
