import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

interface Job {
  id: number;
  title: string;
  description: string;
  job_overview: string;
  responsibilities: string;
  company: string;
  location: string;
  created_by: string;
  created_at: string;
  skills_required: string;
  benefits: string;
  job_type: string;
  qualification_level: string;
  featured?: boolean;
  isFeaturedShown?: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class JobsService {
  private baseUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  getJobs(): Observable<Job[]> {
    return this.http.get<Job[]>(`${this.baseUrl}/jobs/`);
  }
getJobById(id: number): Observable<Job> {
  return this.http.get<Job>(`${this.baseUrl}/jobs/${id}/`);
}
}