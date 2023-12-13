from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import OptionSerializer
import os
import ast
from dotenv import load_dotenv

load_dotenv()


class IndexAPIView(APIView):
    def get(self, request, format=None):
        options = ast.literal_eval(os.getenv("OPTIONS", "[]"))
        serializer = OptionSerializer({'options': options}, context={'request': request})
        return Response(serializer.data, status=status.HTTP_200_OK)
