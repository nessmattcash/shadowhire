"""
URL configuration for analyzer project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.urls import path
from core.views import RegisterView , JobListView, JobCreateView , JobDetailView , RecruiterApprovalView
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from core.views import ResumeUploadView
from django.conf import settings
from django.conf.urls.static import static




urlpatterns = [
    path('api/register/', RegisterView.as_view(), name='register'),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
        path('admin/', admin.site.urls),
        
        path('api/resume/upload/', ResumeUploadView.as_view(), name='resume_upload'),
        path('api/jobs/', JobListView.as_view(), name='job_list'),
        path('api/jobs/<int:pk>/', JobDetailView.as_view(), name='job_detail'),
        path('api/jobs/approve/<int:pk>/', RecruiterApprovalView.as_view(), name='recruiter_approval'),
        path('api/jobs/create/', JobCreateView.as_view(), name='job_create'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


