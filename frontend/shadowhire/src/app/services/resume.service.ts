import { Injectable } from '@angular/core';
import { HttpClient, HttpEventType, HttpRequest, HttpResponse, HttpHeaders } from '@angular/common/http';
import { Observable, map } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ResumeService {
  private apiUrl = 'http://localhost:8000/api';
  private staticAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzU0NjgxMzk1LCJpYXQiOjE3NTQ2Nzc3OTUsImp0aSI6IjYxMTc5ODVlZmViMDQzMTViMWI2MzViM2M0NTViMjVkIiwidXNlcl9pZCI6IjYiLCJpc19yZWNydWl0ZXIiOnRydWV9.O1k9aQGYOUMcz3uxhzUa4OVjPlInFCnQjdgSFpuW5qw';

  constructor(private http: HttpClient) {}

  uploadResume(file: File): Observable<{ progress: number } | { response: any }> {
    const formData = new FormData();
    formData.append('file', file, file.name);

    const req = new HttpRequest('POST', `${this.apiUrl}/resume/upload/`, formData, {
      headers: new HttpHeaders({ Authorization: `Bearer ${this.staticAccessToken}` }),
      reportProgress: true,
      responseType: 'json'
    });

    return this.http.request(req).pipe(
      map(event => {
        if (event.type === HttpEventType.UploadProgress && event.total) {
          return { progress: Math.round((100 * event.loaded) / event.total) };
        } else if (event instanceof HttpResponse) {
          return { response: event.body };
        }
        return { progress: 0 };
      })
    );
  }
}