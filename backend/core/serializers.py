from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

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
        print("Token payload:", token.payload)  # Debug: Use token.payload instead of dict(token)
        return token

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'first_name', 'last_name', 'is_recruiter')
        read_only_fields = ('id', 'is_recruiter')

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    password2 = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = User
        fields = ('username', 'email', 'first_name', 'last_name', 'password', 'password2', 'is_recruiter')

    def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError({"password": "Passwords do not match."})
        return attrs

    def create(self, validated_data):
        validated_data.pop('password2')
        user = User.objects.create_user(**validated_data)
        return user