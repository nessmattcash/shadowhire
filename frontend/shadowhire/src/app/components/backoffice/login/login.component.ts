import { Component, OnInit, AfterViewInit, OnDestroy, ElementRef, ViewEncapsulation } from '@angular/core';
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
    "Evaluating skills...",
    "Checking education...",
    "Analyzing achievements...",
    "Calculating final score..."
  ];

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
    this.createParticles();
    this.setupInputFocusEffects();
    this.setupButtonRippleEffects();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  onSubmit(): void {
    if (this.loginForm.invalid) return;

    this.isLoading = true;
    this.isAnalyzing = true;
    this.analysisComplete = false;

    if (this.loginForm.value.rememberMe) {
      localStorage.setItem('shadowhire_email', this.loginForm.value.email);
      localStorage.setItem('shadowhire_password', this.loginForm.value.password);
    } else {
      localStorage.removeItem('shadowhire_email');
      localStorage.removeItem('shadowhire_password');
    }

    let currentMessage = 0;
    const typewriterInterval = setInterval(() => {
      this.analysisMessage = this.statusMessages[currentMessage];
      currentMessage = (currentMessage + 1) % this.statusMessages.length;
    }, 1000);

    setTimeout(() => {
      clearInterval(typewriterInterval);
      this.analysisMessage = 'CV Analysis: Complete (Score: 87/100)';
      this.isAnalyzing = false;
      this.analysisComplete = true;
      this.isLoading = false;

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
    return this.passwordStrength < 40 ? '#ff4757' : this.passwordStrength < 80 ? '#ffa502' : '#2ed573';
  }

  private createParticles(): void {
    const particlesContainer = this.el.nativeElement.querySelector('#particles');
    if (!particlesContainer) {
      console.error('Particles container not found!');
      return;
    }

    // Clear existing particles
    while (particlesContainer.firstChild) {
      particlesContainer.removeChild(particlesContainer.firstChild);
    }

    const particleCount = 50; // Adjusted for better density
    for (let i = 0; i < particleCount; i++) {
      const particle = document.createElement('div');
      particle.classList.add('particle');

      const size = Math.random() * 2 + 1; // 1-3px
      particle.style.width = `${size}px`;
      particle.style.height = `${size}px`;
      particle.style.left = `${Math.random() * 100}%`;
      particle.style.top = `${Math.random() * 100}%`;
      particle.style.animationDuration = `${Math.random() * 10 + 10}s`; // 10-20s
      particle.style.animationDelay = `${Math.random() * 5}s`; // 0-5s

      particlesContainer.appendChild(particle);
    }
  }

  private setupInputFocusEffects(): void {
    const inputs = this.el.nativeElement.querySelectorAll('.form-control');
    inputs.forEach((input: HTMLElement) => {
      input.addEventListener('focus', () => {
        const highlight = input.parentElement?.querySelector('.input-highlight') as HTMLElement | null;
        if (highlight) (highlight as HTMLElement).style.width = '100%';
      });
      input.addEventListener('blur', () => {
        if (!(input as HTMLInputElement).value) {
          const highlight = input.parentElement?.querySelector('.input-highlight') as HTMLElement | null;
          if (highlight) (highlight as HTMLElement).style.width = '0';
        }
      });
    });
  }

  private setupButtonRippleEffects(): void {
    const buttons = this.el.nativeElement.querySelectorAll('.btn-primary');
    buttons.forEach((button: HTMLElement) => {
      button.addEventListener('click', (e: MouseEvent) => {
        e.preventDefault(); // Prevent form submission here if needed
        const rect = button.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const ripple = document.createElement('span');
        ripple.classList.add('ripple');
        ripple.style.left = `${x}px`;
        ripple.style.top = `${y}px`;

        button.appendChild(ripple);
        setTimeout(() => ripple.remove(), 600);
      });
    });
  }
}