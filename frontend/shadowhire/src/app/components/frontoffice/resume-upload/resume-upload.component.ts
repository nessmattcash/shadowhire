import { Component, ElementRef, ViewChild, AfterViewInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HeaderComponent } from '../header/header.component';
import { FooterComponent } from '../footer/footer.component';
import { Chart, ChartConfiguration, ChartData, ChartOptions, registerables } from 'chart.js';
import { ResumeService } from '../../../services/resume.service'; 

declare const AOS: any;
declare const Parallax: any;

@Component({
  selector: 'app-resume-upload',
  templateUrl: './resume-upload.component.html',
  styleUrls: ['./resume-upload.component.scss'],
  standalone: true,
  imports: [CommonModule, HeaderComponent, FooterComponent]
})
export class ResumeUploadComponent implements AfterViewInit, OnDestroy {
  @ViewChild('fileInput') fileInput!: ElementRef;
  @ViewChild('skillsRadarChart') skillsRadarChart!: ElementRef<HTMLCanvasElement>;

  selectedFile: File | null = null;
  isDragging = false;
  isUploading = false;
  uploadProgress = 0;
  showResults = false;
  uploadedFile: { id: number; filename: string; file_url: string; uploaded_at: string; parsed_text: string } | null = null;
  cvScore = 82;
  contentScore = 85;
  formatScore = 75;
  keywordScore = 86;
  activeTab: 'improvements' | 'jobs' | 'skills' = 'improvements';
  matchedJobs = [
    {
      title: 'Senior Frontend Developer',
      company: 'Tech Innovations Inc.',
      location: 'Remote',
      description: 'Develop cutting-edge web applications using Angular and modern JavaScript frameworks.',
      keySkills: ['Angular', 'TypeScript', 'RxJS', 'SCSS'],
      matchScore: 92
    },
    {
      title: 'UI/UX Engineer',
      company: 'Digital Solutions LLC',
      location: 'New York, NY',
      description: 'Create beautiful and functional user interfaces with a focus on user experience.',
      keySkills: ['Figma', 'CSS', 'JavaScript', 'UX Principles'],
      matchScore: 88
    },
    {
      title: 'Full Stack Developer',
      company: 'WebCraft Studios',
      location: 'San Francisco, CA',
      description: 'Build complete web applications from frontend to backend with modern technologies.',
      keySkills: ['Node.js', 'React', 'MongoDB', 'AWS'],
      matchScore: 85
    }
  ];
  topSkills = [
    { name: 'Angular', demand: 'High' },
    { name: 'TypeScript', demand: 'High' },
    { name: 'JavaScript', demand: 'High' },
    { name: 'HTML/CSS', demand: 'Medium' },
    { name: 'RxJS', demand: 'Medium' }
  ];
  recommendedSkills = [
    { name: 'React', reason: 'Complementary to your Angular skills' },
    { name: 'Node.js', reason: 'Expand into full-stack development' },
    { name: 'AWS', reason: 'High demand for cloud skills' }
  ];
  private chart: Chart | null = null;
  private parallax: any | null = null;

  constructor(private resumeService: ResumeService, private el: ElementRef) {
    Chart.register(...registerables);
  }

  ngAfterViewInit() {
    if (typeof window !== 'undefined') {
      this.initializeAOS();
      this.initializeParallax();
      if (this.showResults && this.skillsRadarChart) {
        this.initializeRadarChart();
      }
    }
  }

  ngOnDestroy() {
    if (this.chart) {
      this.chart.destroy();
    }
    if (this.parallax) {
      this.parallax.destroy();
    }
    if (typeof AOS !== 'undefined') {
      AOS.refresh();
    }
  }

  initializeAOS() {
    if (typeof AOS !== 'undefined') {
      try {
        AOS.init({
          duration: 800,
          easing: 'ease-in-out',
          once: true,
          mirror: false
        });
      } catch (error) {
        console.error('AOS initialization failed:', error);
      }
    } else {
      console.warn('AOS library not loaded');
    }
  }

  initializeParallax() {
    const scene = document.querySelector('.floating-cards');
    if (scene && typeof Parallax !== 'undefined') {
      try {
        this.parallax = new Parallax(scene, {
          relativeInput: true,
          hoverOnly: true,
          calibrateX: true,
          calibrateY: true,
          invertX: false,
          invertY: false,
          limitX: 20,
          limitY: 20
        });
      } catch (error) {
        console.error('Parallax initialization failed:', error);
      }
    } else {
      console.warn('Floating cards element or Parallax library not found');
    }
  }

  initializeRadarChart() {
    if (this.skillsRadarChart) {
      const ctx = this.skillsRadarChart.nativeElement.getContext('2d');
      if (ctx) {
        const data: ChartData<'radar'> = {
          labels: this.topSkills.map(skill => skill.name),
          datasets: [
            {
              label: 'Your Skill Level',
              data: [90, 85, 80, 75, 70],
              backgroundColor: 'rgba(237, 137, 54, 0.2)',
              borderColor: '#ed8936',
              borderWidth: 2,
              pointBackgroundColor: '#ed8936',
              pointBorderColor: '#fff',
              pointHoverBackgroundColor: '#fff',
              pointHoverBorderColor: '#ed8936'
            },
            {
              label: 'Industry Average',
              data: [70, 65, 75, 80, 60],
              backgroundColor: 'rgba(203, 213, 225, 0.2)',
              borderColor: '#cbd5e1',
              borderWidth: 2,
              pointBackgroundColor: '#cbd5e1',
              pointBorderColor: '#fff',
              pointHoverBackgroundColor: '#fff',
              pointHoverBorderColor: '#cbd5e1'
            }
          ]
        };

        const options: ChartOptions<'radar'> = {
          responsive: true,
          scales: {
            r: {
              angleLines: { color: 'rgba(203, 213, 225, 0.1)' },
              grid: { color: 'rgba(203, 213, 225, 0.1)' },
              suggestedMin: 50,
              suggestedMax: 100,
              pointLabels: {
                color: '#e2e8f0',
                font: { family: 'Roboto', size: 12 }
              },
              ticks: {
                backdropColor: 'transparent',
                color: '#94a3b8',
                font: { family: 'Roboto' }
              }
            }
          },
          plugins: {
            legend: {
              labels: {
                color: '#e2e8f0',
                font: { family: 'Roboto', size: 14 },
                padding: 20
              }
            }
          },
          elements: {
            line: { tension: 0.1 }
          }
        };

        this.chart = new Chart(ctx, {
          type: 'radar',
          data,
          options
        } as ChartConfiguration<'radar'>);
      }
    }
  }

  onDragOver(event: DragEvent) {
    event.preventDefault();
    this.isDragging = true;
  }

  onDragLeave(event: DragEvent) {
    event.preventDefault();
    this.isDragging = false;
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    this.isDragging = false;
    if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
      this.handleFile(event.dataTransfer.files[0]);
    }
  }

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.handleFile(input.files[0]);
    }
  }

  handleFile(file: File) {
    const validTypes = ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
    if (!validTypes.includes(file.type)) {
      alert('Please upload a valid file type (PDF, DOC, or DOCX)');
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      alert('File size should not exceed 5MB');
      return;
    }
    this.selectedFile = file;
    this.showResults = false;
  }

  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

 uploadFile() {
    if (!this.selectedFile) {
      alert('Please select a file');
      return;
    }

    this.isUploading = true;
    this.uploadProgress = 0;

    this.resumeService.uploadResume(this.selectedFile).subscribe({
      next: (event) => {
        if ('progress' in event) {
          this.uploadProgress = event.progress;
        } else if ('response' in event) {
          this.isUploading = false;
          this.uploadedFile = event.response;
          this.showResults = true;
          console.log('Upload successful:', event.response);
          if (this.skillsRadarChart) {
            this.initializeRadarChart();
          }
          setTimeout(() => {
            const resultsSection = document.querySelector('.cv-results-section');
            if (resultsSection) {
              resultsSection.scrollIntoView({ behavior: 'smooth' });
            }
          }, 100);
        }
      },
      error: (error) => {
        this.isUploading = false;
        this.uploadProgress = 0;
        const errorMsg = error.error?.file?.[0] || 'Upload failed. Please try again.';
        alert(errorMsg);
        console.error('Upload error:', error);
      },
      complete: () => {
        this.isUploading = false;
      }
    });
  }

  removeFile() {
    this.selectedFile = null;
    this.isUploading = false;
    this.uploadProgress = 0;
    this.showResults = false;
    if (this.fileInput) {
      this.fileInput.nativeElement.value = '';
    }
  }
}