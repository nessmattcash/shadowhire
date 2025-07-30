import { Component, OnInit, AfterViewInit, OnDestroy, ElementRef, ViewEncapsulation, HostListener } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { Subject } from 'rxjs';
import { takeUntil, debounceTime } from 'rxjs/operators';
import { CommonModule } from '@angular/common';

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
  analysisMessage = 'CV Analysis: Ready';
  
  private destroy$ = new Subject<void>();
  private statusMessages = [
    "Scanning your experience...",
    "Evaluating skills matrix...",
    "Checking education background...",
    "Analyzing professional achievements...",
    "Calculating final score..."
  ];
  private particles: Particle[] = [];
  private trails: Trail[] = [];
  private connections: Connection[] = [];
  private mouse = { x: -1000, y: -1000 };
  private animationFrameId: number | null = null;
  private particleCount = 60; // Optimized for cinematic effect
  private connectionDistance = 200;
  private trailLength = 15;
  private lastTrailTime = 0;
  private trailInterval = 50; // ms between trail creation

  constructor(
    private fb: FormBuilder,
    private router: Router,
    private el: ElementRef
  ) {
    this.loginForm = this.fb.group({
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(8)]],
      rememberMe: [false]
    });
  }

  ngOnInit(): void {
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

  @HostListener('mousemove', ['$event'])
  onMouseMove(event: MouseEvent) {
    const particlesContainer = this.el.nativeElement.querySelector('#particles');
    if (particlesContainer) {
      const rect = particlesContainer.getBoundingClientRect();
      this.mouse.x = event.clientX - rect.left;
      this.mouse.y = event.clientY - rect.top;
    }
  }

  onSubmit(): void {
    if (this.loginForm.invalid) return;

    this.isLoading = true;
    this.isAnalyzing = true;
    this.analysisComplete = false;

    // Save credentials if remember me is checked
    if (this.loginForm.value.rememberMe) {
      localStorage.setItem('shadowhire_email', this.loginForm.value.email);
      localStorage.setItem('shadowhire_password', this.loginForm.value.password);
    } else {
      localStorage.removeItem('shadowhire_email');
      localStorage.removeItem('shadowhire_password');
    }

    // Simulate analysis process with cinematic particles
    this.createAnalysisParticles();
    
    let currentMessage = 0;
    const typewriterInterval = setInterval(() => {
      this.analysisMessage = this.statusMessages[currentMessage];
      currentMessage = (currentMessage + 1) % this.statusMessages.length;
      
      // Add more particles during analysis
      if (currentMessage % 2 === 0) {
        this.createAnalysisParticles();
      }
    }, 1000);

    // Complete analysis after 5 seconds
    setTimeout(() => {
      clearInterval(typewriterInterval);
      this.analysisMessage = 'CV Analysis: Complete (Score: 87/100)';
      this.isAnalyzing = false;
      this.analysisComplete = true;
      this.isLoading = false;

      // Create celebration particles
      this.createCelebrationParticles();

      // Navigate to dashboard
      setTimeout(() => this.router.navigate(['/dashboard']), 1000);
    }, 5000);
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

  private initParticles(): void {
    const particlesContainer = this.el.nativeElement.querySelector('#particles');
    if (!particlesContainer) return;

    // Clear existing particles
    particlesContainer.innerHTML = '';

    const containerRect = particlesContainer.getBoundingClientRect();

    for (let i = 0; i < this.particleCount; i++) {
      this.createParticle(particlesContainer, containerRect);
    }
  }

  private createParticle(container: HTMLElement, containerRect: DOMRect): void {
    const particle = document.createElement('div');
    particle.classList.add('particle');

    // Particle properties - cinematic style
    const size = Math.random() * 4 + 2; // 2-6px size
    const x = Math.random() * containerRect.width;
    const y = Math.random() * containerRect.height;
    const speedX = (Math.random() - 0.5) * 0.5;
    const speedY = (Math.random() - 0.5) * 0.5;
    const rotation = Math.random() * 360;
    const hue = 15 + Math.random() * 15; // Orange color range
    const saturation = 80 + Math.random() * 15;
    const lightness = 60 + Math.random() * 20;
    const alpha = 0.7 + Math.random() * 0.2; // More visible
    const color = `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;

    // Set particle styles
    particle.style.width = `${size}px`;
    particle.style.height = `${size}px`;
    particle.style.left = `${x}px`;
    particle.style.top = `${y}px`;
    particle.style.backgroundColor = color;
    particle.style.boxShadow = `0 0 ${size * 4}px ${color}`;
    particle.style.setProperty('--rotation', `${rotation}deg`);
    particle.style.zIndex = '1';

    container.appendChild(particle);

    // Store particle data
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
    
    trail.style.width = `${particle.size * 15}px`;
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
      // Mouse interaction - cinematic repulsion
      const dx = this.mouse.x - particle.x;
      const dy = this.mouse.y - particle.y;
      const distanceToMouse = Math.sqrt(dx * dx + dy * dy);
      const mouseInfluenceRadius = 250;

      if (distanceToMouse < mouseInfluenceRadius) {
        const force = (mouseInfluenceRadius - distanceToMouse) / mouseInfluenceRadius * 2;
        const angle = Math.atan2(dy, dx);
        
        // Move away from mouse with more intensity
        particle.speedX -= Math.cos(angle) * force * 0.8;
        particle.speedY -= Math.sin(angle) * force * 0.8;
        
        // Create more trails when mouse is near
        if (distanceToMouse < mouseInfluenceRadius / 2) {
          this.createTrailSegment(particle);
        }
      }

      // Continuous random movement - smoother
      particle.speedX += (Math.random() - 0.5) * 0.03;
      particle.speedY += (Math.random() - 0.5) * 0.03;

      // Boundary checks with bounce and trail creation
      if (particle.x <= 0 || particle.x >= containerRect.width) {
        particle.speedX = -particle.speedX * 0.7;
        particle.x = particle.x <= 0 ? 1 : containerRect.width - 1;
        this.createTrailSegment(particle);
      }
      
      if (particle.y <= 0 || particle.y >= containerRect.height) {
        particle.speedY = -particle.speedY * 0.7;
        particle.y = particle.y <= 0 ? 1 : containerRect.height - 1;
        this.createTrailSegment(particle);
      }

      // Apply friction for smooth movement
      particle.speedX *= 0.99;
      particle.speedY *= 0.99;

      // Update position
      particle.x += particle.speedX;
      particle.y += particle.speedY;

      // Apply new position with smooth transition
      particle.element.style.transform = `translate(${particle.x}px, ${particle.y}px) rotate(${now * 0.01 % 360}deg)`;
      
      // Dynamic glow based on speed - more dramatic
      const speed = Math.sqrt(particle.speedX * particle.speedX + particle.speedY * particle.speedY);
      const glowIntensity = Math.min(1.5, speed * 4);
      particle.element.style.boxShadow = `0 0 ${particle.size * 4 * (1 + glowIntensity)}px ${particle.color}`;
      
      // Add trail segment periodically
      if (now % 2 === 0) {
        this.createTrailSegment(particle);
      }
    });
  }

  private updateTrails(): void {
    // Update and fade out trails
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
    const particlesContainer = this.el.nativeElement.querySelector('#particles');
    if (!particlesContainer) return;

    // Clear old connections
    this.connections.forEach(conn => conn.element.remove());
    this.connections = [];

    // Create new connections between nearby particles
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
          
          // Position between particles
          const x = (p1.x + p2.x) / 2;
          const y = (p1.y + p2.y) / 2;
          const angle = Math.atan2(dy, dx) * 180 / Math.PI;
          const opacity = 0.3 * (1 - distance/this.connectionDistance);
          
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
    const particlesContainer = this.el.nativeElement.querySelector('#particles');
    if (!particlesContainer) return;

    const containerRect = particlesContainer.getBoundingClientRect();
    const count = 5 + Math.floor(Math.random() * 5);

    for (let i = 0; i < count; i++) {
      const particle = document.createElement('div');
      particle.classList.add('particle');

      // Analysis particle properties - more vibrant
      const size = Math.random() * 5 + 3; // 3-8px size
      const x = containerRect.width * 0.3 + Math.random() * containerRect.width * 0.4;
      const y = containerRect.height * 0.6 + Math.random() * containerRect.height * 0.2;
      const speedX = (Math.random() - 0.5) * 2;
      const speedY = (Math.random() - 0.5) * 2;
      const rotation = Math.random() * 360;
      const hue = 15 + Math.random() * 15;
      const saturation = 90 + Math.random() * 10;
      const lightness = 70 + Math.random() * 20;
      const alpha = 0.9 + Math.random() * 0.1;
      const color = `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;

      particle.style.width = `${size}px`;
      particle.style.height = `${size}px`;
      particle.style.left = `${x}px`;
      particle.style.top = `${y}px`;
      particle.style.backgroundColor = color;
      particle.style.boxShadow = `0 0 ${size * 6}px ${color}`;
      particle.style.setProperty('--rotation', `${rotation}deg`);
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
    const count = 20;

    for (let i = 0; i < count; i++) {
      const particle = document.createElement('div');
      particle.classList.add('particle');

      // Celebration particle properties - brighter and faster
      const size = Math.random() * 6 + 4; // 4-10px size
      const x = containerRect.width / 2;
      const y = containerRect.height / 2;
      const speedX = (Math.random() - 0.5) * 6;
      const speedY = (Math.random() - 0.5) * 6;
      const rotation = Math.random() * 360;
      const hue = 15 + Math.random() * 15;
      const saturation = 100;
      const lightness = 80 + Math.random() * 15;
      const alpha = 1;
      const color = `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;

      particle.style.width = `${size}px`;
      particle.style.height = `${size}px`;
      particle.style.left = `${x}px`;
      particle.style.top = `${y}px`;
      particle.style.backgroundColor = color;
      particle.style.boxShadow = `0 0 ${size * 8}px ${color}`;
      particle.style.setProperty('--rotation', `${rotation}deg`);
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