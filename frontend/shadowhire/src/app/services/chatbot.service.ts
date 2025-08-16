import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ChatbotService {
  private isChatOpen = new BehaviorSubject<boolean>(false);
  isChatOpen$ = this.isChatOpen.asObservable();

  toggleChat(): void {
    this.isChatOpen.next(!this.isChatOpen.value);
  }

  setChatOpen(isOpen: boolean): void {
    this.isChatOpen.next(isOpen);
  }
}