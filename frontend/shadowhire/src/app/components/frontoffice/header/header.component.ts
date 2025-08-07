import { Component, AfterViewInit, OnDestroy, Renderer2, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.scss'],
  standalone: true,
  imports: [CommonModule],
})
export class HeaderComponent implements AfterViewInit, OnDestroy {
  private scrollListener: any;
  private resizeListener: any;
  private mouseMoveListener: any;
  private particles: HTMLElement[] = [];
  private lastScrollPosition = 0;

  constructor(private renderer: Renderer2, private el: ElementRef) {}

  ngAfterViewInit() {
    this.initMobileNav();
    this.initScrollBehavior();
    this.initDropdowns();
    this.initParticles();
    this.initLinkHoverEffects();
    this.initButtonAnimation();
    this.initSmoothScroll();
  }

  ngOnDestroy() {
    if (this.scrollListener) this.scrollListener();
    if (this.resizeListener) this.resizeListener();
    if (this.mouseMoveListener) this.mouseMoveListener();
  }

  private debounce(func: Function, wait: number) {
    let timeout: any;
    return (...args: any[]) => {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  }

  private initMobileNav() {
    const mobileNavToggle = this.el.nativeElement.querySelector('.mobile-nav-toggle');
    const navMenu = this.el.nativeElement.querySelector('#navmenu');

    if (mobileNavToggle && navMenu) {
      this.renderer.listen(mobileNavToggle, 'click', () => {
        const isActive = navMenu.classList.toggle('navmenu-active');
        mobileNavToggle.classList.toggle('bi-list');
        mobileNavToggle.classList.toggle('bi-x');
        this.renderer.setAttribute(mobileNavToggle, 'aria-expanded', isActive.toString());
      });

      const navLinks = this.el.nativeElement.querySelectorAll('.nav-link');
      navLinks.forEach((link: HTMLElement) => {
        this.renderer.listen(link, 'click', () => {
          navMenu.classList.remove('navmenu-active');
          mobileNavToggle.classList.add('bi-list');
          mobileNavToggle.classList.remove('bi-x');
          this.renderer.setAttribute(mobileNavToggle, 'aria-expanded', 'false');
        });
      });
      this.renderer.setAttribute(mobileNavToggle, 'aria-label', 'Toggle navigation');
    }
  }

  private initScrollBehavior() {
    this.scrollListener = this.renderer.listen(window, 'scroll', this.debounce(() => {
      const header = this.el.nativeElement.querySelector('#header');
      const currentScroll = window.pageYOffset;
      const body = document.body;

      if (currentScroll > 100) {
        this.renderer.addClass(header, 'header-shrink');
        this.renderer.setStyle(body, 'padding-top', '60px');
      } else {
        this.renderer.removeClass(header, 'header-shrink');
        this.renderer.setStyle(body, 'padding-top', '80px');
      }

      const logo = this.el.nativeElement.querySelector('.logo-main');
      if (logo) {
        const logoOffset = Math.min(currentScroll * 0.3, 50);
        this.renderer.setStyle(logo, 'transform', `translateY(${logoOffset}px)`);
      }

      if (currentScroll > this.lastScrollPosition && currentScroll > 100) {
        this.renderer.addClass(header, 'header-hide');
      } else {
        this.renderer.removeClass(header, 'header-hide');
      }
      this.lastScrollPosition = Math.max(0, currentScroll);

      this.updateActiveNavLink();
    }, 50));

    this.debounce(() => {
      const header = this.el.nativeElement.querySelector('#header');
      const currentScroll = window.pageYOffset;
      const body = document.body;

      if (currentScroll > 100) {
        this.renderer.addClass(header, 'header-shrink');
        this.renderer.setStyle(body, 'padding-top', '60px');
      } else {
        this.renderer.removeClass(header, 'header-shrink');
        this.renderer.setStyle(body, 'padding-top', '80px');
      }
    }, 50)();
  }

  private updateActiveNavLink() {
    const sections = document.querySelectorAll('section');
    const navLinks = this.el.nativeElement.querySelectorAll('.nav-link');
    let current = '';

    sections.forEach(section => {
      const sectionTop = section.offsetTop;
      const sectionHeight = section.offsetHeight;
      if (window.scrollY >= sectionTop - 200 && window.scrollY < sectionTop + sectionHeight - 200) {
        current = section.getAttribute('id') || '';
      }
    });

    navLinks.forEach((link: HTMLElement) => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  }

  private initSmoothScroll() {
    const links = this.el.nativeElement.querySelectorAll('.nav-link');
    links.forEach((link: HTMLElement) => {
      this.renderer.listen(link, 'click', (e: Event) => {
        e.preventDefault();
        const targetId = link.getAttribute('href')?.substring(1);
        const targetElement = document.getElementById(targetId || '');
        if (targetElement) {
          window.scrollTo({
            top: targetElement.offsetTop - 80,
            behavior: 'smooth',
          });
        }
      });
    });
  }

  private initDropdowns() {
    const dropdowns = this.el.nativeElement.querySelectorAll('.dropdown > a');
    dropdowns.forEach((dropdown: HTMLElement) => {
      this.renderer.setAttribute(dropdown, 'aria-haspopup', 'true');
      this.renderer.setAttribute(dropdown, 'aria-expanded', 'false');
      this.renderer.listen(dropdown, 'click', (e) => {
        e.preventDefault();
        const parent = dropdown.parentElement;
        if (parent) {
          const isActive = !parent.classList.contains('dropdown-active');
          this.el.nativeElement.querySelectorAll('.dropdown').forEach((d: HTMLElement) => {
            if (d !== parent) {
              d.classList.remove('dropdown-active');
              d.querySelector('a')?.setAttribute('aria-expanded', 'false');
            }
          });
          parent.classList.toggle('dropdown-active', isActive);
          this.renderer.setAttribute(dropdown, 'aria-expanded', isActive.toString());
        }
      });
    });

    const submenus = this.el.nativeElement.querySelectorAll('.dropdown-submenu > a');
    submenus.forEach((submenu: HTMLElement) => {
      this.renderer.setAttribute(submenu, 'aria-haspopup', 'true');
      this.renderer.setAttribute(submenu, 'aria-expanded', 'false');
      this.renderer.listen(submenu, 'click', (e) => {
        e.preventDefault();
        const parent = submenu.parentElement;
        if (parent) {
          const isActive = !parent.classList.contains('dropdown-submenu-active');
          this.el.nativeElement.querySelectorAll('.dropdown-submenu').forEach((s: HTMLElement) => {
            if (s !== parent) {
              s.classList.remove('dropdown-submenu-active');
              s.querySelector('a')?.setAttribute('aria-expanded', 'false');
            }
          });
          parent.classList.toggle('dropdown-submenu-active', isActive);
          this.renderer.setAttribute(submenu, 'aria-expanded', isActive.toString());
        }
      });
    });

    this.renderer.listen(document, 'click', (e: Event) => {
      const target = e.target as HTMLElement;
      if (!this.el.nativeElement.contains(target)) {
        this.el.nativeElement.querySelectorAll('.dropdown').forEach((d: HTMLElement) => {
          d.classList.remove('dropdown-active');
          d.querySelector('a')?.setAttribute('aria-expanded', 'false');
        });
        this.el.nativeElement.querySelectorAll('.dropdown-submenu').forEach((s: HTMLElement) => {
          s.classList.remove('dropdown-submenu-active');
          s.querySelector('a')?.setAttribute('aria-expanded', 'false');
        });
      }
    });
  }

  private initParticles() {
    const particles = this.el.nativeElement.querySelectorAll('.particle');
    this.particles = Array.from(particles) as HTMLElement[];

    if (window.innerWidth > 1199) {
      this.mouseMoveListener = this.renderer.listen(window, 'mousemove', this.debounce((e: MouseEvent) => {
        this.particles.forEach((particle, index) => {
          const speed = (index + 1) * 0.02;
          const x = (window.innerWidth - e.pageX) * speed;
          const y = (window.innerHeight - e.pageY) * speed;
          this.renderer.setStyle(particle, 'transform', `translate(${x}px, ${y}px)`);
        });
      }, 100));
    }
  }

  private initLinkHoverEffects() {
    const links = this.el.nativeElement.querySelectorAll('.nav-link');
    links.forEach((link: HTMLElement) => {
      this.renderer.listen(link, 'mouseenter', () => {
        const hoverElement = link.querySelector('.nav-hover');
        if (hoverElement) {
          this.renderer.setStyle(hoverElement, 'width', '100%');
        }
      });

      this.renderer.listen(link, 'mouseleave', () => {
        const hoverElement = link.querySelector('.nav-hover');
        if (hoverElement && !link.classList.contains('active')) {
          this.renderer.setStyle(hoverElement, 'width', '0');
        }
      });
    });
  }

  private initButtonAnimation() {
    const button = this.el.nativeElement.querySelector('.btn-getstarted');
    if (button) {
      this.renderer.listen(button, 'mouseenter', () => {
        const liquid = button.querySelector('.liquid');
        if (liquid) {
          this.renderer.addClass(liquid, 'liquid-animate');
        }
      });

      this.renderer.listen(button, 'mouseleave', () => {
        const liquid = button.querySelector('.liquid');
        if (liquid) {
          this.renderer.removeClass(liquid, 'liquid-animate');
        }
      });
    }
  }
}