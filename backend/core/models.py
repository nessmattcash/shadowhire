from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    is_recruiter = models.BooleanField(default=False)  
    is_admin = models.BooleanField(default=False)
    company_name = models.CharField(max_length=100, blank=True, null=True) 
    def __str__(self):
        return f"{self.first_name} {self.last_name} ({self.email})"

class Resume(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='resumes')
    file = models.FileField(upload_to='resumes/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    parsed_text = models.TextField(blank=True, null=True)  # For storing extracted resume text

    def __str__(self):
        return f"Resume for {self.user.username} ({self.uploaded_at})"

class Job(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    company = models.CharField(max_length=100)
    location = models.CharField(max_length=100)
    job_overview = models.TextField(blank=True, null=True)
    responsibilities = models.TextField(blank=True, null=True)
    benefits = models.TextField(blank=True, null=True)
    job_type = models.CharField(max_length=50, choices=[('remote', 'Remote'), ('onsite', 'Onsite'), ('hybrid', 'Hybrid')], default='onsite')
    qualification_level = models.CharField(
        max_length=50,
        choices=[('junior', 'Junior'), ('mid', 'Mid-level'), ('senior', 'Senior')],
        default='mid'
    )
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='jobs')
    created_at = models.DateTimeField(auto_now_add=True)
    skills_required = models.TextField(blank=True, null=True)  # For job matching

    def __str__(self):
        return self.title

class Application(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='applications')
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name='applications')
    resume = models.ForeignKey(Resume, on_delete=models.CASCADE, related_name='applications')
    applied_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=50, default='Pending')  # e.g., Pending, Reviewed, Accepted

    def __str__(self):
        return f"{self.user.username} applied to {self.job.title}"