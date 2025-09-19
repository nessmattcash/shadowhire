// Updated JobDetailsComponent TS - Replace your existing job-details.component.ts with this
import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, NgForm} from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { HeaderComponent } from '../header/header.component';
import { FooterComponent } from '../footer/footer.component';
import { JobsService } from '../../../services/jobs.service';
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

interface Application {
  name: string;
  email: string;
  resume?: File;
  coverLetter: string;
}

@Component({
  selector: 'app-job-details',
  standalone: true,
  imports: [CommonModule, FormsModule, HeaderComponent, FooterComponent],
  templateUrl: './job-details.component.html',
  styleUrls: ['./job-details.component.scss']
})
export class JobDetailsComponent implements OnInit, OnDestroy {
  job: Job | null = null;
  relatedJobs: Job[] = [];
  showApplyModal = false;
  application: Application = { name: '', email: '', coverLetter: '' };
  savedJobs: number[] = [];
  userSkills: string[] = ['JavaScript', 'React', 'Angular']; // Mock user skills for matching

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private jobsService: JobsService
  ) {}

  ngOnInit(): void {
    AOS.init({ duration: 800, easing: 'ease-out' });
    this.loadSavedJobs();
    const jobId = Number(this.route.snapshot.paramMap.get('id'));
    this.fetchJobDetails(jobId);
  }

  ngOnDestroy(): void {}

fetchJobDetails(id: number): void {
  this.jobsService.getJobById(id).subscribe({
    next: (job: Job) => {
      this.job = { ...job, featured: true, isFeaturedShown: this.isJobSaved(job) };
      this.fetchRelatedJobs();
    },
    error: (err) => {
      console.error('Error fetching job details:', err);
      this.router.navigate(['/jobs']);
    }
  });
}

 fetchRelatedJobs(): void {
    if (!this.job) return;
    // Mock related jobs; replace with backend call to /jobs/related/<company>/ or similar
    this.relatedJobs = [
      {
        id: this.job.id + 1,
        title: `Senior ${this.job.title.split(' ')[0]} Developer`,
        description: `Advanced role at ${this.job.company} focusing on ${this.job.skills_required}.`,
        company: this.job.company,
        location: this.job.location === 'Remote' ? 'New York' : this.job.location,
        created_by: this.job.created_by, // Added created_by
        created_at: new Date().toISOString(),
        skills_required: `Advanced ${this.job.skills_required}`
      },
      {
        id: this.job.id + 2,
        title: `Junior ${this.job.title.split(' ')[0]} Specialist`,
        description: `Entry-level position at ${this.job.company} with ${this.job.skills_required}.`,
        company: this.job.company,
        location: 'Remote',
        created_by: this.job.created_by, // Added created_by
        created_at: new Date().toISOString(),
        skills_required: `Basic ${this.job.skills_required}`
      }
    ];
  }

  applyForJob(): void {
    this.showApplyModal = true;
  }

  closeApplyModal(event?: Event): void {
    if (event) event.preventDefault();
    this.showApplyModal = false;
  }

  stopPropagation(event: Event): void {
    event.stopPropagation();
  }

  onFileChange(event: any): void {
    this.application.resume = event.target.files[0];
  }

  submitApplication(form: NgForm): void {
    if (form.valid && this.application.resume) {
      // In a real app, POST to backend /applications/ with job.id, user token, etc.
      console.log('Application submitted:', this.application, 'for job:', this.job?.id);
      alert('Application submitted successfully!');
      this.closeApplyModal();
      form.resetForm();
      this.application = { name: '', email: '', coverLetter: '' };
    } else {
      alert('Please fill all required fields and select a resume.');
    }
  }

  visitCompanyWebsite(): void {
    // Mock; in real, use actual company URL from backend
    window.open('https://example.com/' + this.job?.company?.toLowerCase().replace(/\s+/g, '-'), '_blank');
  }

  navigateToRelatedJob(id: number): void {
    this.router.navigate(['/jobs-details', id]);
  }

  getSkillsArray(skillsString: string): string[] {
    return skillsString.split(',').map(skill => skill.trim());
  }

  isSkillMatched(skill: string): boolean {
    return this.userSkills.some(userSkill => userSkill.toLowerCase() === skill.toLowerCase());
  }

  getResponsibilities(description: string): string[] {
    // Simple mock parsing: split by sentences
    if (!description) return [];
    return description.split('.').slice(0, 5).map(s => s.trim() + '.').filter(s => s.length > 10);
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
    const companyLogos: { [key: string]: string } = {
      'EY': '/assets/EY.jpg',
      'Actia': '/assets/ACTIA.jpg',
      'Sagemcom': '/assets/sagemcom.jpg',
      'Defensy': '/assets/def.png',
      'Verse': '/assets/verse.jpg.png',
      'Sopra Steria': '/assets/sopra.jpg',
      'Orange': '/assets/orange.jpg',
      'Ooredoo': '/assets/ooredoo.jpg',
      'InstaDeep': '/assets/insta.png',
      'RFC': '/assets/rfc.png',
      'Vermeg': '/assets/vermeg.jpg',
      'Expensya': '/assets/exp.jpg',
      'Sofrecom': '/assets/sofrecom.jpg',
      'Talan': '/assets/talan.jpg',
      'NeoSoft': '/assets/neo.jpg',
      'Amaris': '/assets/amaris.jpg',
      'Business & Decision': '/assets/bd.png',
      'Telnet': '/assets/telnet.jpg',
      'Focus': '/assets/focus.jpg',
      'IBM': '/assets/ibm.jpg',
      'Oradist': '/assets/images.jpg',
      'Wattnow': '/assets/wat.png',
      'Dabchy': '/assets/dabchy.png',
      'Roamsmart': '/assets/roam.jpg',
      'Capgemini': '/assets/cap.jpeg'
    };
    return companyLogos[companyName] || '/assets/default-logo.jpg';
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
}