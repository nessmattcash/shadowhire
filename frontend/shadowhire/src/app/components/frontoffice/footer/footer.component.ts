import { Component, OnInit, HostListener, Renderer2, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-footer',
  templateUrl: './footer.component.html',
  styleUrls: ['./footer.component.scss'],
  standalone: true,
  imports: [CommonModule],
})
export class FooterComponent implements OnInit {
  infoEmail = 'contact@shadowhire.com';
  currentYear = new Date().getFullYear();
  private scrollListener: any;

  constructor(private renderer: Renderer2, private el: ElementRef) {}

  ngOnInit() {
    this.initBackToTopButton();
    this.initSocialIconsAnimation();
    this.initFooterLinkAnimations();
  }

  private initBackToTopButton() {
    const backToTop = this.el.nativeElement.querySelector('.back-to-top');
    if (backToTop) {
      this.scrollListener = this.renderer.listen(window, 'scroll', () => {
        if (window.scrollY > 300) {
          this.renderer.addClass(backToTop, 'active');
        } else {
          this.renderer.removeClass(backToTop, 'active');
        }
      });
    }
  }

  private initSocialIconsAnimation() {
    const socialIcons = this.el.nativeElement.querySelectorAll('.social-icon');
    socialIcons.forEach((icon: HTMLElement) => {
      this.renderer.listen(icon, 'mouseenter', () => {
        this.renderer.addClass(icon, 'hover');
      });
      this.renderer.listen(icon, 'mouseleave', () => {
        this.renderer.removeClass(icon, 'hover');
      });
    });
  }

  private initFooterLinkAnimations() {
    const footerLinks = this.el.nativeElement.querySelectorAll('.footer-link');
    footerLinks.forEach((link: HTMLElement) => {
      const arrow = link.querySelector('i');
      
      this.renderer.listen(link, 'mouseenter', () => {
        this.renderer.addClass(link, 'hover');
        if (arrow) {
          this.renderer.setStyle(arrow, 'transform', 'translateX(5px)');
        }
      });
      
      this.renderer.listen(link, 'mouseleave', () => {
        this.renderer.removeClass(link, 'hover');
        if (arrow) {
          this.renderer.setStyle(arrow, 'transform', 'translateX(0)');
        }
      });
    });
  }

  ngOnDestroy() {
    if (this.scrollListener) {
      this.scrollListener();
    }
  }
}