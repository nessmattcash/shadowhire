from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .models import Resume
from .models import Job
from .models import Application

User = get_user_model()

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    # Override the username field to accept 'username' in payload
    username = serializers.CharField()

    def validate(self, attrs):
        print("CustomTokenObtainPairSerializer: validate called with attrs =", attrs)  # Debug
        # Accept 'username' from payload and map to email
        email = attrs.get('username')
        if not email:
            raise serializers.ValidationError({"username": "This field is required."})

        # Map to credentials for authentication
        credentials = {
            'username': email.lower(),
            'password': attrs.get('password')
        }
        print("Credentials passed to parent validate:", credentials)  # Debug
        data = super().validate(credentials)
        print("Validation successful, data =", data)  # Debug
        return data

    @classmethod
    def get_token(cls, user):
        print("CustomTokenObtainPairSerializer: get_token called for user =", user.email)  # Debug
        token = super().get_token(user)
        token['is_recruiter'] = user.is_recruiter
        token['is_admin'] = user.is_admin
        print("Token payload:", token.payload)  # Debug: Use token.payload instead of dict(token)
        return token

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'first_name', 'last_name', 'is_recruiter' , 'is_admin', 'company_name')
        read_only_fields = ('id', 'is_recruiter', 'is_admin', 'company_name')

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    password2 = serializers.CharField(write_only=True, required=True)
    company_name = serializers.CharField(required=False, write_only=True)

    class Meta:
        model = User
        fields = ('username', 'email', 'first_name', 'last_name', 'password', 'password2', 'is_recruiter', 'is_admin', 'company_name')

    def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError({"password": "Passwords do not match."})
        
        is_admin = attrs.get('is_admin', False)
        is_recruiter = attrs.get('is_recruiter', False)
        company_name = attrs.get('company_name')
        
        if is_admin:
            if User.objects.filter(is_admin=True).exists():
                raise serializers.ValidationError({"is_admin": "There can be only one admin."})
            if attrs.get('email') != 'aziz.mehdi@esprit.tn':
                raise serializers.ValidationError({"email": "Admin email must be aziz.mehdi@esprit.tn."})
            if company_name:
                raise serializers.ValidationError({"company_name": "Admin cannot have a company."})
        elif is_recruiter:
            if not company_name:
                raise serializers.ValidationError({"company_name": "Company name is required for recruiters."})
        else:
            if company_name:
                raise serializers.ValidationError({"company_name": "Normal users cannot have a company."})
        
        return attrs

    def create(self, validated_data):
        validated_data.pop('password2')
        company_name = validated_data.pop('company_name', None)
        user = User.objects.create_user(**validated_data)
        if company_name:
            user.company_name = company_name
            user.save()
        return user
class ResumeSerializer(serializers.ModelSerializer):
    file = serializers.FileField()

    class Meta:
        model = Resume
        fields = ['id', 'file',  'uploaded_at', 'parsed_text']
        read_only_fields = ['id', 'uploaded_at', 'parsed_text']

    def validate_file(self, value):
        valid_extensions = ['.pdf', '.doc', '.docx']
        extension = '.' + value.name.rsplit('.', 1)[1].lower() if '.' in value.name else ''
        if extension not in valid_extensions:
            raise serializers.ValidationError("Only PDF, DOC, and DOCX files are allowed.")
        max_size = 5 * 1024 * 1024  # 5MB
        if value.size > max_size:
            raise serializers.ValidationError("File size must be less than 5MB.")
        return value

    def create(self, validated_data):
        user = self.context['request'].user
        validated_data['user'] = user
        validated_data['parsed_text'] = ""  # Placeholder for future parsing
        return super().create(validated_data)  

class JobSerializer(serializers.ModelSerializer):
    created_by = serializers.StringRelatedField(read_only=True)

    class Meta:
        model = Job
        fields = ['id', 'title', 'description', 'company', 'location', 'created_by', 'created_at', 'skills_required']
        read_only_fields = ['created_by', 'created_at', 'company']  # company is set in validate

    def validate(self, attrs):
        user = self.context['request'].user
        if not user.is_recruiter:
            raise serializers.ValidationError({"error": "Only recruiters/admins can create jobs."})
        if not user.is_admin:
            if not user.company_name:
                raise serializers.ValidationError({"company": "Recruiter must have a company name."})
            if 'company' in attrs:
                raise serializers.ValidationError({"company": "Non-admin recruiters cannot specify company."})
            attrs['company'] = user.company_name  # Auto-set for non-admins
        else:
            if 'company' not in attrs:
                raise serializers.ValidationError({"company": "Admins must provide a company name."})
        return attrs

    def create(self, validated_data):
        return Job.objects.create(**validated_data)      