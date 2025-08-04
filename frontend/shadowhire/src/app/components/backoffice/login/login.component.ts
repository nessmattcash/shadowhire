import { Component, OnInit, AfterViewInit, OnDestroy, ElementRef, ViewEncapsulation, HostListener } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { Subject } from 'rxjs';
import { takeUntil, debounceTime, first } from 'rxjs/operators';
import { CommonModule } from '@angular/common';
import { AuthService } from '../../../services/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [ReactiveFormsModule, RouterModule, CommonModule],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss'],
  encapsulation: ViewEncapsulation.None,
})
export class LoginComponent implements OnInit, AfterViewInit, OnDestroy {
  loginForm: FormGroup;
  passwordStrength = 0;
  isLoading = false;
  isAnalyzing = false;
  analysisComplete = false;
  analysisMessage = 'Ready to log in';
  screenSize = 'desktop';
  showPassword = false;
  errorMessage: string | null = null;

  private destroy$ = new Subject<void>();
  private statusMessages = [
    "Scanning credentials...",
    "Verifying user profile...",
    "Checking authentication...",
    "Preparing dashboard...",
    "Access granted!"
  ];
  private particles: Particle[] = [];
  private trails: Trail[] = [];
  private connections: Connection[] = [];
  private mouse = { x: -1000, y: -1000 };
  private animationFrameId: number | null = null;
  private particleCount = 30;
  private connectionDistance = 150;
  private trailLength = 10;
  private lastTrailTime = 0;
  private trailInterval = 100;

  constructor(
    private fb: FormBuilder,
    private router: Router,
    private el: ElementRef,
    private authService: AuthService
  ) {
    this.loginForm = this.fb.group({
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(8)]],
      rememberMe: [false]
    });
  }

  ngOnInit(): void {
    this.checkScreenSize();
    
    const savedEmail = localStorage.getItem('shadowhire_email');
    const savedPassword = localStorage.getItem('shadowhire_password');
    
    if (savedEmail && savedPassword) {
      this.loginForm.patchValue({
        email: savedEmail,
        password: savedPassword,
        rememberMe: true
      });
    }

    this.loginForm.get('password')?.valueChanges
      .pipe(debounceTime(300), takeUntil(this.destroy$))
      .subscribe((password: string) => this.calculatePasswordStrength(password));
  }

  ngAfterViewInit(): void {
    this.initParticles();
    this.startAnimation();
    this.setupMouseMove();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
    this.cleanupParticles();
  }

  togglePasswordVisibility(): void {
    this.showPassword = !this.showPassword;
  }

  calculatePasswordStrength(password: string): void {
    let strength = 0;
    if (password.length > 0) strength += 20;
    if (password.length >= 8) strength += 20;
    if (/[A-Z]/.test(password)) strength += 20;
    if (/[0-9]/.test(password)) strength += 20;
    if (/[^A-Za-z0-9]/.test(password)) strength += 20;
    this.passwordStrength = strength;
  }

  getPasswordStrengthColor(): string {
    if (this.passwordStrength < 40) return '#ff4757';
    if (this.passwordStrength < 80) return '#ffa502';
    return '#2ed573';
  }

  onSubmit(): void {
  if (this.loginForm.invalid) {
    this.loginForm.markAllAsTouched();
    this.errorMessage = 'Please enter a valid email and password.';
    return;
  }

  this.isLoading = true;
  this.errorMessage = null;

  const loginData = {
    username: this.loginForm.value.email.toLowerCase(),
    password: this.loginForm.value.password
  };

  console.log('Login payload:', loginData);

  this.authService.login(loginData).pipe(first()).subscribe({
    next: (response) => {
      if (this.loginForm.value.rememberMe) {
        localStorage.setItem('shadowhire_email', this.loginForm.value.email);
        localStorage.setItem('shadowhire_password', this.loginForm.value.password);
      } else {
        localStorage.removeItem('shadowhire_email');
        localStorage.removeItem('shadowhire_password');
      }

      localStorage.setItem('access_token', response.access);
      localStorage.setItem('refresh_token', response.refresh);

      this.isAnalyzing = true;
      this.analysisComplete = false;
      let currentMessage = 0;
      const typewriterInterval = setInterval(() => {
        this.analysisMessage = this.statusMessages[currentMessage];
        currentMessage = (currentMessage + 1) % this.statusMessages.length;
        if (this.screenSize !== 'mobile' || currentMessage % 2 === 0) {
          this.createAnalysisParticles();
        }
      }, 100);

      setTimeout(() => {
        clearInterval(typewriterInterval);
        this.analysisMessage = 'Login Successful!';
        this.isAnalyzing = false;
        this.analysisComplete = true;
        this.createCelebrationParticles();

        const token = response.access;
        let payload;
        try {
          payload = JSON.parse(atob(token.split('.')[1]));
          console.log('JWT payload:', payload); // Debug full payload
        } catch (e) {
          console.error('Failed to parse JWT payload:', e);
          this.errorMessage = 'Login Denied: Invalid token format.';
          this.isLoading = false;
          return;
        }

        // Log is_recruiter specifically
        console.log('is_recruiter:', payload.is_recruiter);

        const redirectUrl = payload.is_recruiter ? '/candidate-management' : '/';
        this.router.navigate([redirectUrl]).then((success) => {
          if (success) {
            console.log('Navigation successful to:', redirectUrl);
          } else {
            console.error('Navigation failed to:', redirectUrl);
            this.errorMessage = 'Login Denied: Navigation error.';
          }
          this.isLoading = false; // Set after navigation
        });
      }, 500); // Increased to 500ms for smoother animation
    },
    error: (error) => {
      this.isLoading = false;
      let errorMsg = 'Login Denied: Invalid email or password.';
      if (error.error) {
        if (error.error.detail) {
          errorMsg = `Login Denied: ${error.error.detail}`;
        } else if (error.error.non_field_errors) {
          errorMsg = `Login Denied: ${error.error.non_field_errors.join(' ')}`;
        } else if (error.error.username) {
          errorMsg = `Login Denied: ${error.error.username.join(' ')}`;
        } else if (error.error.password) {
          errorMsg = `Login Denied: ${error.error.password.join(' ')}`;
        }
      }
      this.errorMessage = errorMsg;
      console.error('Login failed', error);
    }
  });
}
// Add this to your component
clearError() {
  this.errorMessage = '';
}
  private checkScreenSize(): void {
    const width = window.innerWidth;
    if (width < 768) {
      this.screenSize = 'mobile';
    } else if (width < 992) {
      this.screenSize = 'tablet';
    } else {
      this.screenSize = 'desktop';
    }
    this.adjustParticleEffects();
  }

  @HostListener('window:resize')
  onResize() {
    this.checkScreenSize();
  }

  private adjustParticleEffects(): void {
    switch (this.screenSize) {
      case 'mobile':
        this.particleCount = 15;
        this.connectionDistance = 100;
        break;
      case 'tablet':
        this.particleCount = 20;
        this.connectionDistance = 120;
        break;
      default:
        this.particleCount = 30;
        this.connectionDistance = 150;
    }
    this.cleanupParticles();
    this.initParticles();
  }

  private initParticles(): void {
    const particlesContainer = this.el.nativeElement.querySelector('#particles');
    if (!particlesContainer) return;

    particlesContainer.innerHTML = '';
    const containerRect = particlesContainer.getBoundingClientRect();

    for (let i = 0; i < this.particleCount; i++) {
      this.createParticle(particlesContainer, containerRect);
    }
  }

  private createParticle(container: HTMLElement, containerRect: DOMRect): void {
    const particle = document.createElement('div');
    particle.classList.add('particle');

    const size = Math.random() * 3 + 1;
    const x = Math.random() * containerRect.width;
    const y = Math.random() * containerRect.height;
    const speedX = (Math.random() - 0.5) * 0.3;
    const speedY = (Math.random() - 0.5) * 0.3;
    const hue = 15 + Math.random() * 15;
    const saturation = 80 + Math.random() * 15;
    const lightness = 60 + Math.random() * 20;
    const alpha = 0.5 + Math.random() * 0.3;
    const color = `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;

    particle.style.width = `${size}px`;
    particle.style.height = `${size}px`;
    particle.style.left = `${x}px`;
    particle.style.top = `${y}px`;
    particle.style.backgroundColor = color;
    particle.style.boxShadow = `0 0 ${size * 2}px ${color}`;
    particle.style.zIndex = '1';

    container.appendChild(particle);

    this.particles.push({
      element: particle,
      x,
      y,
      size,
      speedX,
      speedY,
      color,
      trail: []
    });
  }

  private createTrailSegment(particle: Particle): void {
    const now = Date.now();
    if (now - this.lastTrailTime < this.trailInterval) return;
    this.lastTrailTime = now;

    const particlesContainer = this.el.nativeElement.querySelector('#particles');
    if (!particlesContainer) return;

    const trail = document.createElement('div');
    trail.classList.add('particle-trail');
    
    trail.style.width = `${particle.size * 10}px`;
    trail.style.height = `${particle.size}px`;
    trail.style.left = `${particle.x}px`;
    trail.style.top = `${particle.y}px`;
    trail.style.backgroundColor = particle.color;
    trail.style.opacity = '0';
    trail.style.transform = `rotate(${Math.atan2(particle.speedY, particle.speedX) * 180 / Math.PI}deg)`;
    
    particlesContainer.appendChild(trail);
    
    this.trails.push({
      element: trail,
      x: particle.x,
      y: particle.y,
      size: particle.size,
      opacity: 0,
      lifespan: 0
    });
  }

  private startAnimation(): void {
    const animate = () => {
      this.updateParticles();
      this.updateTrails();
      this.drawConnections();
      this.animationFrameId = requestAnimationFrame(animate);
    };
    animate();
  }

  private updateParticles(): void {
    const particlesContainer = this.el.nativeElement.querySelector('#particles');
    if (!particlesContainer) return;

    const containerRect = particlesContainer.getBoundingClientRect();
    const now = Date.now();

    this.particles.forEach((particle) => {
      if (this.screenSize === 'desktop') {
        const dx = this.mouse.x - particle.x;
        const dy = this.mouse.y - particle.y;
        const distanceToMouse = Math.sqrt(dx * dx + dy * dy);
        const mouseInfluenceRadius = 200;

        if (distanceToMouse < mouseInfluenceRadius) {
          const force = (mouseInfluenceRadius - distanceToMouse) / mouseInfluenceRadius * 1.5;
          const angle = Math.atan2(dy, dx);
          
          particle.speedX -= Math.cos(angle) * force * 0.6;
          particle.speedY -= Math.sin(angle) * force * 0.6;
          
          if (distanceToMouse < mouseInfluenceRadius / 2) {
            this.createTrailSegment(particle);
          }
        }
      }

      const randomFactor = this.screenSize === 'mobile' ? 0.01 : 0.02;
      particle.speedX += (Math.random() - 0.5) * randomFactor;
      particle.speedY += (Math.random() - 0.5) * randomFactor;

      if (particle.x <= 0 || particle.x >= containerRect.width) {
        particle.speedX = -particle.speedX * 0.7;
        particle.x = particle.x <= 0 ? 1 : containerRect.width - 1;
        if (this.screenSize !== 'mobile') this.createTrailSegment(particle);
      }
      
      if (particle.y <= 0 || particle.y >= containerRect.height) {
        particle.speedY = -particle.speedY * 0.7;
        particle.y = particle.y <= 0 ? 1 : containerRect.height - 1;
        if (this.screenSize !== 'mobile') this.createTrailSegment(particle);
      }

      particle.speedX *= 0.98;
      particle.speedY *= 0.98;

      particle.x += particle.speedX;
      particle.y += particle.speedY;

      particle.element.style.transform = `translate(${particle.x}px, ${particle.y}px)`;
      
      if (this.screenSize !== 'mobile') {
        const speed = Math.sqrt(particle.speedX * particle.speedX + particle.speedY * particle.speedY);
        const glowIntensity = Math.min(1, speed * 3);
        particle.element.style.boxShadow = `0 0 ${particle.size * 3 * (1 + glowIntensity)}px ${particle.color}`;
      }
    });
  }

  private updateTrails(): void {
    this.trails.forEach((trail, index) => {
      trail.lifespan += 1;
      trail.opacity = 1 - (trail.lifespan / this.trailLength);
      
      if (trail.opacity <= 0) {
        trail.element.remove();
        this.trails.splice(index, 1);
      } else {
        trail.element.style.opacity = `${trail.opacity}`;
        trail.element.style.transform = `scale(${0.3 + trail.opacity * 0.7})`;
      }
    });
  }

  private drawConnections(): void {
    if (this.screenSize === 'mobile') return;

    const particlesContainer = this.el.nativeElement.querySelector('#particles');
    if (!particlesContainer) return;

    this.connections.forEach(conn => conn.element.remove());
    this.connections = [];

    for (let i = 0; i < this.particles.length; i++) {
      for (let j = i + 1; j < this.particles.length; j++) {
        const p1 = this.particles[i];
        const p2 = this.particles[j];
        
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < this.connectionDistance) {
          const connection = document.createElement('div');
          connection.classList.add('particle-connection');
          
          const x = (p1.x + p2.x) / 2;
          const y = (p1.y + p2.y) / 2;
          const angle = Math.atan2(dy, dx) * 180 / Math.PI;
          const opacity = 0.2 * (1 - distance/this.connectionDistance);
          
          connection.style.width = `${distance}px`;
          connection.style.left = `${x - distance/2}px`;
          connection.style.top = `${y}px`;
          connection.style.transform = `rotate(${angle}deg)`;
          connection.style.opacity = `${opacity}`;
          connection.style.background = `linear-gradient(to right, 
            transparent 0%, 
            rgba(255, 107, 53, ${opacity}) 50%, 
            transparent 100%)`;
          
          particlesContainer.appendChild(connection);
          
          this.connections.push({
            element: connection,
            p1: i,
            p2: j
          });
        }
      }
    }
  }

  private createAnalysisParticles(): void {
    if (this.screenSize === 'mobile' && Math.random() > 0.3) return;

    const particlesContainer = this.el.nativeElement.querySelector('#particles');
    if (!particlesContainer) return;

    const containerRect = particlesContainer.getBoundingClientRect();
    const count = this.screenSize === 'mobile' ? 2 : 3 + Math.floor(Math.random() * 4);

    for (let i = 0; i < count; i++) {
      const particle = document.createElement('div');
      particle.classList.add('particle');

      const size = Math.random() * 4 + 2;
      const x = containerRect.width * 0.3 + Math.random() * containerRect.width * 0.4;
      const y = containerRect.height * 0.6 + Math.random() * containerRect.height * 0.2;
      const speedX = (Math.random() - 0.5) * 1.5;
      const speedY = (Math.random() - 0.5) * 1.5;
      const hue = 15 + Math.random() * 15;
      const saturation = 90 + Math.random() * 10;
      const lightness = 70 + Math.random() * 20;
      const alpha = 0.7 + Math.random() * 0.2;
      const color = `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;

      particle.style.width = `${size}px`;
      particle.style.height = `${size}px`;
      particle.style.left = `${x}px`;
      particle.style.top = `${y}px`;
      particle.style.backgroundColor = color;
      particle.style.boxShadow = `0 0 ${size * 4}px ${color}`;
      particle.style.zIndex = '1';

      particlesContainer.appendChild(particle);

      this.particles.push({
        element: particle,
        x,
        y,
        size,
        speedX,
        speedY,
        color,
        trail: []
      });
    }
  }

  private createCelebrationParticles(): void {
    const particlesContainer = this.el.nativeElement.querySelector('#particles');
    if (!particlesContainer) return;

    const containerRect = particlesContainer.getBoundingClientRect();
    const count = this.screenSize === 'mobile' ? 10 : 15;

    for (let i = 0; i < count; i++) {
      const particle = document.createElement('div');
      particle.classList.add('particle');

      const size = Math.random() * 5 + 3;
      const x = containerRect.width / 2;
      const y = containerRect.height / 2;
      const speedX = (Math.random() - 0.5) * 4;
      const speedY = (Math.random() - 0.5) * 4;
      const hue = 15 + Math.random() * 15;
      const saturation = 100;
      const lightness = 80 + Math.random() * 15;
      const alpha = 0.9;
      const color = `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;

      particle.style.width = `${size}px`;
      particle.style.height = `${size}px`;
      particle.style.left = `${x}px`;
      particle.style.top = `${y}px`;
      particle.style.backgroundColor = color;
      particle.style.boxShadow = `0 0 ${size * 6}px ${color}`;
      particle.style.zIndex = '1';

      particlesContainer.appendChild(particle);

      this.particles.push({
        element: particle,
        x,
        y,
        size,
        speedX,
        speedY,
        color,
        trail: []
      });
    }
  }

  private cleanupParticles(): void {
    const particlesContainer = this.el.nativeElement.querySelector('#particles');
    if (particlesContainer) {
      particlesContainer.innerHTML = '';
    }
    this.particles = [];
    this.trails = [];
    this.connections = [];
  }

  private setupMouseMove(): void {
    if (this.screenSize === 'mobile') return;

    this.el.nativeElement.addEventListener('mousemove', (event: MouseEvent) => {
      const particlesContainer = this.el.nativeElement.querySelector('#particles');
      if (particlesContainer) {
        const rect = particlesContainer.getBoundingClientRect();
        this.mouse.x = event.clientX - rect.left;
        this.mouse.y = event.clientY - rect.top;
      }
    });

    this.el.nativeElement.addEventListener('mouseout', () => {
      this.mouse.x = -1000;
      this.mouse.y = -1000;
    });
  }
}

interface Particle {
  element: HTMLElement;
  x: number;
  y: number;
  size: number;
  speedX: number;
  speedY: number;
  color: string;
  trail: number[];
}

interface Trail {
  element: HTMLElement;
  x: number;
  y: number;
  size: number;
  opacity: number;
  lifespan: number;
}

interface Connection {
  element: HTMLElement;
  p1: number;
  p2: number;
}