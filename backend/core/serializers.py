from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .models import Resume, Job, Application

User = get_user_model()

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    username = serializers.CharField()

    def validate(self, attrs):
        print("CustomTokenObtainPairSerializer: validate called with attrs =", attrs)  # Debug
        email = attrs.get('username')
        if not email:
            raise serializers.ValidationError({"username": "This field is required."})

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
        print("Token payload:", token.payload)  # Debug
        return token

class UserSerializer(serializers.ModelSerializer):
    specialties = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = (
            'id', 'username', 'email', 'first_name', 'last_name', 'is_recruiter', 'is_admin', 'company_name',
            'profile_photo', 'specialties', 'role_in_company', 'company_website', 'company_logo',
            'company_location', 'verification_document', 'is_approved'
        )
        read_only_fields = ('id', 'is_approved')  # is_recruiter, is_admin, company_name set in RegisterSerializer

    def get_specialties(self, obj):
        # Return specialties as list for API response
        if obj.specialties:
            return obj.specialties.split(',')
        return []

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    password2 = serializers.CharField(write_only=True, required=True)
    company_name = serializers.CharField(required=False, write_only=True)
    profile_photo = serializers.ImageField(required=False, write_only=True)
    specialties = serializers.MultipleChoiceField(
        choices=[
            ("Software Engineer", "Software Engineer"),
            ("Web Developer", "Web Developer"),
            ("Front-end Developer", "Front-end Developer"),
            ("Back-end Developer", "Back-end Developer"),
            ("Full Stack Developer", "Full Stack Developer"),
            ("Mobile Developer (iOS/Android)", "Mobile Developer (iOS/Android)"),
            ("Data Scientist", "Data Scientist"),
            ("Data Engineer", "Data Engineer"),
            ("DevOps Engineer", "DevOps Engineer"),
            ("Cloud Engineer (AWS/Azure/GCP)", "Cloud Engineer (AWS/Azure/GCP)"),
            ("AI/ML Engineer", "AI/ML Engineer"),
            ("Cybersecurity Analyst", "Cybersecurity Analyst"),
            ("Network Engineer", "Network Engineer"),
            ("Database Administrator", "Database Administrator"),
            ("UI/UX Designer", "UI/UX Designer"),
            ("QA/Test Engineer", "QA/Test Engineer"),
            ("Systems Administrator", "Systems Administrator"),
            ("Embedded Systems Engineer", "Embedded Systems Engineer"),
            ("Game Developer", "Game Developer"),
            ("Blockchain Developer", "Blockchain Developer"),
            ("IoT Engineer", "IoT Engineer"),
            ("ERP Consultant", "ERP Consultant"),
            ("Business Intelligence Analyst", "Business Intelligence Analyst"),
            ("Robotics Engineer", "Robotics Engineer"),
            ("AR/VR Developer", "AR/VR Developer")
        ],
        required=False, write_only=True
    )
    role_in_company = serializers.ChoiceField(
        choices=[
            ('hr_manager', 'HR Manager'),
            ('talent_acquisition_specialist', 'Talent Acquisition Specialist'),
            ('recruiter', 'Recruiter'),
            ('hiring_manager', 'Hiring Manager'),
            ('other', 'Other')
        ],
        required=False, write_only=True
    )
    company_website = serializers.URLField(required=False, write_only=True)
    company_logo = serializers.ImageField(required=False, write_only=True)
    company_location = serializers.CharField(max_length=100, required=False, write_only=True)
    verification_document = serializers.FileField(required=False, write_only=True)

    class Meta:
        model = User
        fields = (
            'username', 'email', 'first_name', 'last_name', 'password', 'password2', 'is_recruiter', 'is_admin',
            'company_name', 'profile_photo', 'specialties', 'role_in_company', 'company_website',
            'company_logo', 'company_location', 'verification_document'
        )

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
            if company_name or attrs.get('specialties') or attrs.get('role_in_company') or attrs.get('company_website') or attrs.get('company_logo') or attrs.get('company_location') or attrs.get('verification_document'):
                raise serializers.ValidationError({"error": "Admin cannot have company, specialties, or recruiter fields."})
        elif is_recruiter:
            required_fields = ['company_name', 'role_in_company', 'company_website', 'company_location', 'verification_document']
            for field in required_fields:
                if not attrs.get(field):
                    raise serializers.ValidationError({field: f"{field.replace('_', ' ').title()} is required for recruiters."})
            if attrs.get('specialties'):
                raise serializers.ValidationError({"specialties": "Recruiters cannot have specialties."})
        else:  # Regular user
            if company_name or attrs.get('role_in_company') or attrs.get('company_website') or attrs.get('company_logo') or attrs.get('company_location') or attrs.get('verification_document'):
                raise serializers.ValidationError({"error": "Regular users cannot have recruiter fields."})
        
        return attrs

    def create(self, validated_data):
        validated_data.pop('password2')
        company_name = validated_data.pop('company_name', None)
        specialties = validated_data.pop('specialties', None)
        role_in_company = validated_data.pop('role_in_company', None)
        company_website = validated_data.pop('company_website', None)
        company_logo = validated_data.pop('company_logo', None)
        company_location = validated_data.pop('company_location', None)
        verification_document = validated_data.pop('verification_document', None)
        profile_photo = validated_data.pop('profile_photo', None)

        user = User.objects.create_user(**validated_data)
        
        if user.is_recruiter:
            user.company_name = company_name
            user.role_in_company = role_in_company
            user.company_website = company_website
            user.company_logo = company_logo
            user.company_location = company_location
            user.verification_document = verification_document
            user.is_approved = False  # Pending approval
        elif not user.is_admin:
            if specialties:
                user.specialties = ','.join(specialties)  # Store as comma-separated string
        
        if profile_photo:
            user.profile_photo = profile_photo
        
        user.save()
        return user

class ResumeSerializer(serializers.ModelSerializer):
    file = serializers.FileField()

    class Meta:
        model = Resume
        fields = ['id', 'file', 'uploaded_at', 'parsed_text']
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
        fields = [
            'id', 'title', 'description', 'job_overview', 'responsibilities', 'benefits',
            'company', 'location', 'created_by', 'created_at', 'skills_required', 'job_type', 'qualification_level'
        ]
        read_only_fields = ['created_by', 'created_at', 'company']

    def validate(self, attrs):
        user = self.context['request'].user
        if not user.is_recruiter and not user.is_admin:
            raise serializers.ValidationError({"error": "Only recruiters or admins can create jobs."})
        if user.is_recruiter and not user.is_approved:
            raise serializers.ValidationError({"error": "Your recruiter account is pending approval."})
        if not user.is_admin:
            if not user.company_name:
                raise serializers.ValidationError({"company": "Recruiter must have a company name."})
            if 'company' in attrs:
                raise serializers.ValidationError({"company": "Non-admin recruiters cannot specify company."})
            attrs['company'] = user.company_name
        else:
            if 'company' not in attrs:
                raise serializers.ValidationError({"company": "Admins must provide a company name."})
        return attrs

    def create(self, validated_data):
        return Job.objects.create(**validated_data)