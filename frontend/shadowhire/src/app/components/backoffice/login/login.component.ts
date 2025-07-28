import { Component, OnInit, AfterViewInit, OnDestroy, ElementRef, Renderer2 } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { Subject, takeUntil, debounceTime } from 'rxjs';
import { ReactiveFormsModule } from '@angular/forms';
import { RouterModule } from '@angular/router';
import { ViewEncapsulation } from '@angular/core';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss'],
  imports: [ReactiveFormsModule, RouterModule], // Add ReactiveFormsModule and RouterModule
  encapsulation: ViewEncapsulation.None, // Disable encapsulation
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
    private el: ElementRef,
    private renderer: Renderer2
  ) {
    this.loginForm = this.fb.group({
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(8)]],
      rememberMe: [false]
    });
  }

  ngOnInit(): void {
    // Check for saved credentials if "remember me" was checked previously
    const savedEmail = localStorage.getItem('shadowhire_email');
    const savedPassword = localStorage.getItem('shadowhire_password');
    
    if (savedEmail && savedPassword) {
      this.loginForm.patchValue({
        email: savedEmail,
        password: savedPassword,
        rememberMe: true
      });
    }

    // Password strength calculation
    this.loginForm.get('password')?.valueChanges
      .pipe(
        debounceTime(300),
        takeUntil(this.destroy$)
      )
      .subscribe(password => {
        this.calculatePasswordStrength(password);
      });
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

    // Save credentials if "remember me" is checked
    if (this.loginForm.value.rememberMe) {
      localStorage.setItem('shadowhire_email', this.loginForm.value.email);
      localStorage.setItem('shadowhire_password', this.loginForm.value.password);
    } else {
      localStorage.removeItem('shadowhire_email');
      localStorage.removeItem('shadowhire_password');
    }

    // Simulate CV analysis process
    let currentMessage = 0;
    const typewriterInterval = setInterval(() => {
      this.analysisMessage = this.statusMessages[currentMessage];
      currentMessage = (currentMessage + 1) % this.statusMessages.length;
    }, 1000);

    // Final result after delay
    setTimeout(() => {
      clearInterval(typewriterInterval);
      this.analysisMessage = 'CV Analysis: Complete (Score: 87/100)';
      this.isAnalyzing = false;
      this.analysisComplete = true;
      this.isLoading = false;

      // Redirect after analysis
      setTimeout(() => {
        this.router.navigate(['/dashboard']);
      }, 1000);
    }, 5000);
  }

  calculatePasswordStrength(password: string): void {
    let strength = 0;
    
    // Length check
    if (password.length > 0) strength += 20;
    if (password.length >= 8) strength += 20;
    
    // Complexity checks
    if (/[A-Z]/.test(password)) strength += 20;
    if (/[0-9]/.test(password)) strength += 20;
    if (/[^A-Za-z0-9]/.test(password)) strength += 20;
    
    this.passwordStrength = strength;
  }

  getPasswordStrengthColor(): string {
    if (this.passwordStrength < 40) {
      return '#ff4757'; // Red
    } else if (this.passwordStrength < 80) {
      return '#ffa502'; // Orange
    } else {
      return '#2ed573'; // Green
    }
  }

  private createParticles(): void {
    const particlesContainer = this.el.nativeElement.querySelector('#particles');
    const particleCount = 30;
    
    for (let i = 0; i < particleCount; i++) {
      const particle = this.renderer.createElement('div');
      this.renderer.addClass(particle, 'particle');
      
      // Random size between 1px and 3px
      const size = Math.random() * 2 + 1;
      this.renderer.setStyle(particle, 'width', `${size}px`);
      this.renderer.setStyle(particle, 'height', `${size}px`);
      
      // Random position
      this.renderer.setStyle(particle, 'left', `${Math.random() * 100}%`);
      this.renderer.setStyle(particle, 'top', `${Math.random() * 100}%`);
      
      // Random animation duration (10s to 20s)
      const duration = Math.random() * 10 + 10;
      this.renderer.setStyle(particle, 'animationDuration', `${duration}s`);
      
      // Random delay
      this.renderer.setStyle(particle, 'animationDelay', `${Math.random() * 5}s`);
      
      this.renderer.appendChild(particlesContainer, particle);
    }
  }

  private setupInputFocusEffects(): void {
    const inputs = this.el.nativeElement.querySelectorAll('.form-control');
    
    inputs.forEach((input: HTMLElement) => {
      this.renderer.listen(input, 'focus', () => {
        const highlight = input.parentElement?.querySelector('.input-highlight');
        if (highlight) {
          this.renderer.setStyle(highlight, 'width', '100%');
        }
      });
      
      this.renderer.listen(input, 'blur', () => {
        if (!input.getAttribute('value')) {
          const highlight = input.parentElement?.querySelector('.input-highlight');
          if (highlight) {
            this.renderer.setStyle(highlight, 'width', '0');
          }
        }
      });
    });
  }

  private setupButtonRippleEffects(): void {
    const buttons = this.el.nativeElement.querySelectorAll('.btn-primary');
    
    buttons.forEach((button: HTMLElement) => {
      this.renderer.listen(button, 'click', (e: MouseEvent) => {
        const rect = button.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const ripple = this.renderer.createElement('span');
        this.renderer.addClass(ripple, 'ripple');
        this.renderer.setStyle(ripple, 'left', `${x}px`);
        this.renderer.setStyle(ripple, 'top', `${y}px`);
        
        this.renderer.appendChild(button, ripple);
        
        setTimeout(() => {
          this.renderer.removeChild(button, ripple);
        }, 600);
      });
    });
  }
}