import { Component, OnInit, OnDestroy, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterLink } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { HeaderComponent } from '../header/header.component';
import { FooterComponent } from '../footer/footer.component';
import { JobsService } from '../../../services/jobs.service';
import Swiper from 'swiper';
import { Autoplay, EffectFade } from 'swiper/modules';
import AOS from 'aos';

interface Job {
  id: number;
  title: string;
  description: string;
  company: string;
  location: string;
  created_by: string;
  created_at: string;
  skills_required: string;
  featured?: boolean;
  isFeaturedShown?: boolean;
}

@Component({
  selector: 'app-jobs',
  standalone: true,
  imports: [CommonModule, RouterLink, FormsModule, HeaderComponent, FooterComponent],
  templateUrl: './jobs.component.html',
  styleUrls: ['./jobs.component.scss']
})
export class JobsComponent implements OnInit, OnDestroy, AfterViewInit {
  @ViewChild('companySwiper') companySwiperRef!: ElementRef;
  private companySwiper!: Swiper;

  jobs: Job[] = [];
  filteredJobs: Job[] = [];
  isLoading = true;
  error: string | null = null;
  searchTerm = '';
  skillsFilter = '';
  locationFilter = '';
  companyFilter = '';
  currentPage = 1;
  savedJobs: number[] = [];
  selectedJob: Job | null = null;
  progress = 0;
  backgroundImages = [
    '/assets/EY.jpg',
    '/assets/micro.jpg',
    '/assets/OpenAI.jpg'
  ];

  constructor(private jobsService: JobsService, private router: Router) {}

  ngOnInit(): void {
    AOS.init({ duration: 800, easing: 'ease-out' });
    this.loadSavedJobs();
    this.fetchJobs();
    this.simulateProgress();
  }

  ngAfterViewInit(): void {
    if (this.companySwiperRef) {
      this.initSwiper();
    }
  }

  ngOnDestroy(): void {
    if (this.companySwiper) {
      this.companySwiper.destroy();
    }
  }

  private initSwiper(): void {
    this.companySwiper = new Swiper(this.companySwiperRef.nativeElement, {
      modules: [Autoplay, EffectFade],
      loop: true,
      speed: 1500,
      autoplay: {
        delay: 10000,
        disableOnInteraction: false,
      },
      effect: 'slide',
      slidesPerView: 1,
      centeredSlides: true,
      grabCursor: true,
      on: {
        slideChange: () => {
          if (this.companySwiper) {
            const activeIndex = this.companySwiper.realIndex % this.backgroundImages.length;
            this.updateBackground(this.backgroundImages[activeIndex]);
          }
        },
      },
    });
    if (this.companySwiper) {
      this.updateBackground(this.backgroundImages[0]);
    }
  }

  private updateBackground(image: string): void {
    document.documentElement.style.setProperty('--background-image', `url('${image}')`);
  }

  private simulateProgress(): void {
    if (this.isLoading) {
      const interval = setInterval(() => {
        this.progress = Math.min(this.progress + 10, 90);
        if (!this.isLoading) {
          this.progress = 100;
          clearInterval(interval);
        }
      }, 300);
    }
  }

  fetchJobs(): void {
    this.isLoading = true;
    this.progress = 0;
    this.error = null;

    this.jobsService.getJobs().subscribe({
      next: (data: Job[]) => {
        this.jobs = data.map(job => ({ ...job, featured: true, isFeaturedShown: false }));
        this.applyFilters();
        this.isLoading = false;
      },
      error: (err: any) => {
        this.error = 'Failed to load jobs. Please try again later.';
        this.isLoading = false;
        console.error('Error fetching jobs:', err);
      }
    });
  }

  onSearchChange(event: any): void {
    this.searchTerm = event.target.value;
    this.applyFilters();
  }

  triggerSearch(): void {
    this.applyFilters();
  }

  applyFilters(): void {
    let results = [...this.jobs];

    if (this.searchTerm) {
      const term = this.searchTerm.toLowerCase();
      results = results.filter(job =>
        job.title.toLowerCase().includes(term) ||
        job.company.toLowerCase().includes(term) ||
        job.skills_required.toLowerCase().includes(term) ||
        job.description.toLowerCase().includes(term)
      );
    }

    if (this.skillsFilter) {
      const skills = this.skillsFilter.toLowerCase().split(',').map(s => s.trim());
      results = results.filter(job =>
        skills.some(skill => job.skills_required.toLowerCase().includes(skill))
      );
    }

    if (this.locationFilter) {
      results = results.filter(job =>
        job.location.toLowerCase().includes(this.locationFilter.toLowerCase())
      );
    }

    if (this.companyFilter) {
      results = results.filter(job =>
        job.company.toLowerCase().includes(this.companyFilter.toLowerCase())
      );
    }

    this.filteredJobs = results;
    this.currentPage = 1;
  }

  clearFilters(): void {
    this.searchTerm = '';
    this.skillsFilter = '';
    this.locationFilter = '';
    this.companyFilter = '';
    this.applyFilters();
  }

  openJobPopup(job: Job): void {
    this.selectedJob = { ...job };
    const header = document.querySelector('app-header');
    if (header) header.classList.add('hidden');
    setTimeout(() => {
      const popup = document.querySelector('.popup-content');
      if (popup) popup.classList.add('show');
    }, 50);
  }

  closeJobPopup(event?: Event): void {
    if (event) event.preventDefault();
    const popup = document.querySelector('.popup-content');
    if (popup) popup.classList.remove('show');
    const header = document.querySelector('app-header');
    if (header) header.classList.remove('hidden');
    setTimeout(() => {
      this.selectedJob = null;
    }, 300);
  }

  stopPropagation(event: Event): void {
    event.stopPropagation();
  }

  selectJob(job: Job): void {
    this.selectedJob = this.selectedJob === job ? null : job;
  }

  isJobSelected(job: Job): boolean {
    return this.selectedJob === job;
  }

  getSkillsArray(skillsString: string): string[] {
    return skillsString.split(',').map(skill => skill.trim()).slice(0, 4);
  }

  formatDate(dateString: string): string {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - date.getTime());
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;

    return date.toLocaleDateString();
  }

  getCompanyLogo(companyName: string): string {
    return '/assets/default-logo.jpg'; // Default logo
  }

  loadSavedJobs(): void {
    const saved = localStorage.getItem('savedJobs');
    if (saved) {
      this.savedJobs = JSON.parse(saved);
    }
  }

  toggleSaveJob(job: Job): void {
    const index = this.savedJobs.indexOf(job.id);
    if (index === -1) {
      this.savedJobs.push(job.id);
      job.isFeaturedShown = true;
    } else {
      this.savedJobs.splice(index, 1);
      job.isFeaturedShown = false;
    }
    localStorage.setItem('savedJobs', JSON.stringify(this.savedJobs));
  }

  isJobSaved(job: Job): boolean {
    return this.savedJobs.includes(job.id);
  }

  getPageNumbers(): number[] {
    return Array.from({ length: Math.ceil(this.filteredJobs.length / 6) }, (_, i) => i + 1);
  }

  goToPage(page: number): void {
    this.currentPage = page;
  }

  get paginatedJobs(): Job[] {
    const start = (this.currentPage - 1) * 6;
    const end = start + 6;
    return this.filteredJobs.slice(start, end);
  }

  getTwoLineDescription(description: string): string {
    const sentences = description.split('. ').slice(0, 2).join('. ');
    return sentences + (sentences.length < description.length ? '…' : '');
  }

  navigateToJob(jobId: number): void {
    this.router.navigate(['/jobs-details', jobId]).then(() => {
    }).catch(err => console.error('Navigation error:', err));
  }

  getDescriptionHeight(description: string): string {
    const lines = description.split('\n').length || 1;
    return `${Math.min(lines * 1.6, 10)}em`;
  }
}