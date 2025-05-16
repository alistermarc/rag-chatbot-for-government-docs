from django.urls import path
from .views import ChatAPIView, CheckAPIView, FileUploadAPIView

urlpatterns = [
    path('chat/', ChatAPIView.as_view(), name='chat'),
    path('check/', CheckAPIView.as_view(), name='check'), 
    path('upload/', FileUploadAPIView.as_view(), name='upload')
]
