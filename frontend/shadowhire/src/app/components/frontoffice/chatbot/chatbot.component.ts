import { Component, OnInit, OnDestroy, AfterViewInit, ViewChild, ElementRef } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError, Subscription } from 'rxjs';
import { catchError, finalize } from 'rxjs/operators';
import { ChatbotService } from '../../../services/chatbot.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AnimationBuilder, animate, style } from '@angular/animations';
import { environment } from '../../../../environments/environment';

interface ChatMessage {
  sender: 'user' | 'bot';
  text: string;
  time: Date;
}

@Component({
  selector: 'app-chatbot',
  templateUrl: './chatbot.component.html',
  styleUrls: ['./chatbot.component.scss'],
  standalone: true,
  imports: [CommonModule, FormsModule]
})
export class ChatbotComponent implements OnInit, OnDestroy, AfterViewInit {
  @ViewChild('chatMessages') private chatMessagesContainer!: ElementRef;

  userInput: string = '';
  chatHistory: ChatMessage[] = [];
  isLoading: boolean = false;
  isChatOpen: boolean = false;
  isInitialized: boolean = false;
  showInitialQuestions: boolean = true;
  errorMessage: string = '';
  private apiUrl = environment.production ? environment.chatbotUrl : 'http://localhost:5000/chat';
  private subscription: Subscription = new Subscription();
  initialQuestions: string[] = [
    'What is Shadowhire?',
    'How does job matching work?',
    'How accurate is the resume score?'
  ];

  constructor(
    private http: HttpClient,
    private chatbotService: ChatbotService,
    private animationBuilder: AnimationBuilder,
    private el: ElementRef
  ) {
    this.subscription.add(
      this.chatbotService.isChatOpen$.subscribe(isOpen => {
        this.isChatOpen = isOpen;
        this.showInitialQuestions = isOpen;
        if (isOpen && !this.isInitialized) {
          this.isInitialized = true;
          setTimeout(() => this.animateChatOpening(), 50);
        }
      })
    );
  }

  ngOnInit(): void {
    const savedChat = localStorage.getItem('shadowhire_chat_history');
    if (savedChat) {
      this.chatHistory = JSON.parse(savedChat);
    }
  }

  ngAfterViewInit(): void {
    this.scrollToBottom();
  }

  ngOnDestroy(): void {
    this.subscription.unsubscribe();
  }

  animateChatOpening(): void {
    const animation = this.animationBuilder.build([
      style({ transform: 'translateY(20px)', opacity: 0 }),
      animate('300ms ease-out', style({ transform: 'translateY(0)', opacity: 1 }))
    ]);
    const player = animation.create(this.el.nativeElement.querySelector('.chatbot-window'));
    player.play();
  }

  animateMessage(element: HTMLElement, isBot: boolean): void {
    const animation = this.animationBuilder.build([
      style({ transform: isBot ? 'translateX(-10px)' : 'translateX(10px)', opacity: 0 }),
      animate('200ms ease-out', style({ transform: 'translateX(0)', opacity: 1 }))
    ]);
    const player = animation.create(element);
    player.play();
  }

  toggleChat(): void {
    this.chatbotService.toggleChat();
    this.errorMessage = '';
    if (this.isChatOpen) {
      this.scrollToBottom();
    }
  }

  selectInitialQuestion(question: string): void {
    this.userInput = question;
    this.sendMessage();
    this.showInitialQuestions = false;
  }

  sendMessage(): void {
    if (!this.userInput.trim()) {
      this.errorMessage = 'Please enter a question.';
      return;
    }

    this.isLoading = true;
    this.errorMessage = '';
    this.showInitialQuestions = false;

    const question = this.userInput.trim();
    const userMessage: ChatMessage = {
      sender: 'user',
      text: question,
      time: new Date()
    };

    this.chatHistory.push(userMessage);
    this.saveChatHistory();
    this.scrollToBottom();

    this.http
      .post<{ answer: string; error?: string }>(this.apiUrl, { question })
      .pipe(
        catchError(this.handleError),
        finalize(() => {
          this.isLoading = false;
          this.scrollToBottom();
        })
      )
      .subscribe({
        next: (response) => {
          if (response.error) {
            this.errorMessage = response.error;
            return;
          }
          const botMessage: ChatMessage = {
            sender: 'bot',
            text: response.answer,
            time: new Date()
          };
          this.chatHistory.push(botMessage);
          this.saveChatHistory();
          setTimeout(() => {
            const messages = this.chatMessagesContainer.nativeElement.querySelectorAll('.message');
            if (messages.length > 0) {
              this.animateMessage(messages[messages.length - 1], true);
            }
          }, 10);
        },
        error: (error) => {
          this.errorMessage = error.message;
        }
      });

    this.userInput = '';
  }

  private saveChatHistory(): void {
    localStorage.setItem('shadowhire_chat_history', JSON.stringify(this.chatHistory));
  }

  private scrollToBottom(): void {
    setTimeout(() => {
      if (this.chatMessagesContainer) {
        this.chatMessagesContainer.nativeElement.scrollTop =
          this.chatMessagesContainer.nativeElement.scrollHeight;
      }
    }, 100);
  }

  private handleError(error: HttpErrorResponse): Observable<never> {
    let errorMessage = 'An error occurred. Please try again.';
    if (error.status === 400) {
      errorMessage = 'Invalid question format.';
    } else if (error.status === 500) {
      errorMessage = 'Server error. Please try later.';
    } else if (error.error?.error) {
      errorMessage = error.error.error;
    }
    return throwError(() => new Error(errorMessage));
  }

  clearChat(): void {
    this.chatHistory = [];
    this.errorMessage = '';
    this.userInput = '';
    this.showInitialQuestions = true;
    localStorage.removeItem('shadowhire_chat_history');
    this.scrollToBottom();
  }
}