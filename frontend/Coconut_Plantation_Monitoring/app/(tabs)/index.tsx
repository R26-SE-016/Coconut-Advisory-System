import React, { useState, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  SafeAreaView,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Image } from 'expo-image';
import * as Location from 'expo-location';
import { determineSeason, getMonthName, determineZone } from '../../utils/contextHelper';

import { API_ENDPOINTS } from '../../constants/Api';

// Constants
const BACKEND_URL = API_ENDPOINTS.ASK;


// Types
type Source = {
  title: string;
  content: string;
};

type Message = {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  sources?: Source[];
  context_used?: string;
};

export default function AdvisoryChatScreen() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! I am CocoCastAI, your AI Agricultural Expert. Ask me anything about coconut farming. How can I help you today?',
      sender: 'ai',
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isInitializing, setIsInitializing] = useState(true);
  const [userContext, setUserContext] = useState<string | null>(null);
  const flatListRef = useRef<FlatList>(null);

  React.useEffect(() => {
    (async () => {
      let { status } = await Location.requestForegroundPermissionsAsync();

      const currentDate = new Date();
      const currentSeason = determineSeason(currentDate);
      const currentMonth = getMonthName(currentDate);

      let currentZone = 'Unknown Zone';
      if (status === 'granted') {
        try {
          let location = await Location.getCurrentPositionAsync({});
          currentZone = determineZone(location.coords.latitude, location.coords.longitude);
        } catch (error) {
          console.log("Error getting location", error);
        }
      }

      const contextString = `${currentZone} | ${currentSeason} Season (${currentMonth})`;
      setUserContext(contextString);

      // Simulate a small delay for better UX with the logo
      setTimeout(() => {
        setIsInitializing(false);
      }, 2000);
    })();
  }, []);

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText.trim(),
      sender: 'user',
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const response = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: userMessage.text, context: userContext }),
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data = await response.json();

      if (data.success) {
        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: data.answer,
          sender: 'ai',
          sources: data.sources,
          context_used: data.context_used,
        };
        setMessages((prev) => [...prev, aiMessage]);
      } else {
        throw new Error(data.error || 'Failed to get an answer.');
      }
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: `Sorry, I couldn't reach the server. Please check your connection to the backend. (${error.message})`,
        sender: 'ai',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const renderMessage = ({ item }: { item: Message }) => {
    const isUser = item.sender === 'user';

    return (
      <View style={[styles.messageWrapper, isUser ? styles.messageWrapperUser : styles.messageWrapperAi]}>
        {!isUser && (
          <View style={styles.avatarContainer}>
            <Image
              source={require('../../assets/images/logo.png')}
              style={styles.avatarImage}
            />
          </View>
        )}
        <View style={[styles.messageBubble, isUser ? styles.userBubble : styles.aiBubble]}>
          <Text style={[styles.messageText, isUser ? styles.userText : styles.aiText]}>
            {item.text}
          </Text>

          {item.context_used && (
            <View style={styles.contextUsedContainer}>
              <Text style={styles.contextUsedText}>
                <Ionicons name="location-outline" size={11} /> Context Used: {item.context_used}
              </Text>
            </View>
          )}

          {item.sources && item.sources.length > 0 && (
            <View style={styles.sourcesContainer}>
              <Text style={styles.sourceHeader}>Sources:</Text>
              {item.sources.map((src, index) => (
                <Text key={index} style={styles.sourceText} numberOfLines={1}>
                  • {src.title}
                </Text>
              ))}
            </View>
          )}
        </View>
      </View>
    );
  };

  if (isInitializing) {
    return (
      <View style={styles.loadingContainer}>
        <Image
          source={require('../../assets/images/logo.png')}
          style={styles.loadingLogo}
          contentFit="contain"
        />
        <ActivityIndicator size="large" color="#2E7D32" style={styles.loadingIndicator} />
        <Text style={styles.loadingText}>Initializing CocoCastAI...</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerTitleContainer}>
          <Ionicons name="leaf-outline" size={28} color="#FFFFFF" style={styles.headerIcon} />
          <Text style={styles.headerTitle}>CocoCastAI</Text>
        </View>
        <Text style={styles.headerSubtitle}>Your Agricultural Expert</Text>
      </View>

      {/* Context Banner */}
      {userContext && (
        <View style={styles.contextBanner}>
          <Ionicons name="location" size={16} color="#2E7D32" />
          <Text style={styles.contextBannerText}>{userContext}</Text>
        </View>
      )}

      <KeyboardAvoidingView
        style={styles.keyboardAvoid}
        behavior={Platform.OS === 'ios' ? 'padding' : 'padding'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}
      >
        {/* Chat List */}
        <FlatList
          ref={flatListRef}
          data={messages}
          keyExtractor={(item) => item.id}
          renderItem={renderMessage}
          contentContainerStyle={styles.chatContainer}
          onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
          onLayout={() => flatListRef.current?.scrollToEnd({ animated: true })}
        />

        {/* Input Area */}
        <View style={styles.inputContainer}>
          <TextInput
            style={styles.textInput}
            value={inputText}
            onChangeText={setInputText}
            placeholder="Ask about coconut farming..."
            placeholderTextColor="#888"
            multiline
            maxLength={500}
          />
          <TouchableOpacity
            style={[styles.sendButton, (!inputText.trim() || isLoading) && styles.sendButtonDisabled]}
            onPress={sendMessage}
            disabled={!inputText.trim() || isLoading}
          >
            {isLoading ? (
              <ActivityIndicator size="small" color="#FFFFFF" />
            ) : (
              <Ionicons name="send" size={20} color="#FFFFFF" />
            )}
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F9F6', // Very light green/white
  },
  header: {
    backgroundColor: '#2E7D32', // Deep nature green
    paddingTop: Platform.OS === 'android' ? 40 : 20,
    paddingBottom: 20,
    paddingHorizontal: 20,
    borderBottomLeftRadius: 24,
    borderBottomRightRadius: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 12,
    elevation: 8,
    zIndex: 10,
  },
  headerTitleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 4,
  },
  headerIcon: {
    marginRight: 8,
  },
  headerTitle: {
    color: '#FFFFFF',
    fontSize: 24,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  headerSubtitle: {
    color: '#E8F5E9',
    fontSize: 14,
    textAlign: 'center',
    fontWeight: '500',
    opacity: 0.9,
  },
  keyboardAvoid: {
    flex: 1,
  },
  chatContainer: {
    padding: 16,
    paddingBottom: 32,
  },
  messageWrapper: {
    flexDirection: 'row',
    marginBottom: 16,
    alignItems: 'flex-end',
  },
  messageWrapperUser: {
    justifyContent: 'flex-end',
  },
  messageWrapperAi: {
    justifyContent: 'flex-start',
  },
  avatarContainer: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#E8F5E9',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 8,
  },
  messageBubble: {
    maxWidth: '80%',
    padding: 16,
    borderRadius: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  userBubble: {
    backgroundColor: '#2E7D32',
    borderBottomRightRadius: 4,
  },
  aiBubble: {
    backgroundColor: '#FFFFFF',
    borderBottomLeftRadius: 4,
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  messageText: {
    fontSize: 16,
    lineHeight: 24,
  },
  userText: {
    color: '#FFFFFF',
  },
  aiText: {
    color: '#333333',
  },
  contextBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#E8F5E9',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#C8E6C9',
  },
  contextBannerText: {
    marginLeft: 8,
    color: '#2E7D32',
    fontSize: 14,
    fontWeight: '600',
  },
  contextUsedContainer: {
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#F0F0F0',
  },
  contextUsedText: {
    fontSize: 11,
    color: '#2E7D32',
    fontStyle: 'italic',
  },
  sourcesContainer: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#F0F0F0',
  },
  sourceHeader: {
    fontSize: 12,
    fontWeight: '700',
    color: '#666666',
    marginBottom: 4,
  },
  sourceText: {
    fontSize: 12,
    color: '#888888',
    fontStyle: 'italic',
    marginTop: 2,
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 16,
    backgroundColor: '#FFFFFF',
    borderTopWidth: 1,
    borderTopColor: '#EEEEEE',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 4,
  },
  textInput: {
    flex: 1,
    backgroundColor: '#F5F5F5',
    borderRadius: 24,
    paddingHorizontal: 20,
    paddingTop: 14,
    paddingBottom: 14,
    fontSize: 16,
    maxHeight: 120,
    color: '#333',
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  sendButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#2E7D32',
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 12,
    shadowColor: '#2E7D32',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 6,
    elevation: 4,
  },
  sendButtonDisabled: {
    backgroundColor: '#A5D6A7',
    shadowOpacity: 0,
    elevation: 0,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
  },
  loadingLogo: {
    width: 200,
    height: 200,
    marginBottom: 20,
  },
  loadingIndicator: {
    marginBottom: 10,
  },
  loadingText: {
    color: '#2E7D32',
    fontSize: 16,
    fontWeight: '600',
  },
  avatarImage: {
    width: 28,
    height: 28,
    borderRadius: 14,
  },
});
