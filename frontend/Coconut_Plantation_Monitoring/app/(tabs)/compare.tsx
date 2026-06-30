import React, { useState } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  SafeAreaView,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Location from 'expo-location';
import { determineSeason, getMonthName, determineZone } from '../../utils/contextHelper';

import { API_ENDPOINTS } from '../../constants/Api';

const BACKEND_URL = API_ENDPOINTS.COMPARE;


type ComparisonResult = {
  plain_llm: { answer: string };
  rag_system: { answer: string; sources?: { title: string; content: string }[]; context_used?: string };
};

export default function CompareScreen() {
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ComparisonResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [userContext, setUserContext] = useState<string | null>(null);

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
    })();
  }, []);

  const compareAnswers = async () => {
    if (!inputText.trim()) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: inputText.trim(), context: userContext }),
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data = await response.json();

      if (data.success) {
        setResult({
          plain_llm: data.plain_llm,
          rag_system: data.rag_system,
        });
      } else {
        throw new Error(data.error || 'Failed to get an answer.');
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred during comparison.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerTitleContainer}>
          <Ionicons name="git-compare" size={28} color="#FFFFFF" style={styles.headerIcon} />
          <Text style={styles.headerTitle}>LLM vs RAG</Text>
        </View>
        <Text style={styles.headerSubtitle}>See the difference verified context makes</Text>
      </View>

      {/* Context Banner */}
      {userContext && (
        <View style={styles.contextBanner}>
          <Ionicons name="location" size={16} color="#1E3A8A" />
          <Text style={styles.contextBannerText}>{userContext}</Text>
        </View>
      )}

      <KeyboardAvoidingView
        style={styles.keyboardAvoid}
        behavior={Platform.OS === 'ios' ? 'padding' : 'padding'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}
      >
        <ScrollView contentContainerStyle={styles.scrollContainer}>
          {error && (
            <View style={styles.errorContainer}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          )}

          {!result && !isLoading && !error && (
            <View style={styles.emptyState}>
              <Ionicons name="chatbubbles-outline" size={64} color="#ccc" />
              <Text style={styles.emptyText}>Ask a highly specific farming question to compare the AI models.</Text>
            </View>
          )}

          {isLoading && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#2E7D32" />
              <Text style={styles.loadingText}>Running both models...</Text>
            </View>
          )}

          {result && !isLoading && (
            <View style={styles.resultsContainer}>
              {/* Plain LLM Card */}
              <View style={[styles.card, styles.plainCard]}>
                <View style={[styles.cardHeader, styles.plainHeader]}>
                  <Ionicons name="warning-outline" size={20} color="#E65100" />
                  <Text style={styles.plainHeaderText}>Plain LLM (No Context)</Text>
                </View>
                <View style={styles.cardBody}>
                  <Text style={styles.answerText}>{result.plain_llm.answer}</Text>
                  <Text style={styles.warningNote}>⚠️ May contain hallucinations</Text>
                </View>
              </View>

              {/* RAG System Card */}
              <View style={[styles.card, styles.ragCard]}>
                <View style={[styles.cardHeader, styles.ragHeader]}>
                  <Ionicons name="checkmark-circle-outline" size={20} color="#1B5E20" />
                  <Text style={styles.ragHeaderText}>CocoCastAI (Verified)</Text>
                </View>
                <View style={styles.cardBody}>
                  <Text style={styles.answerText}>{result.rag_system.answer}</Text>
                  
                  {result.rag_system.context_used && (
                    <View style={styles.contextUsedContainer}>
                      <Text style={styles.contextUsedText}>
                        <Ionicons name="location-outline" size={12} /> Context Used: {result.rag_system.context_used}
                      </Text>
                    </View>
                  )}
                  
                  {result.rag_system.sources && result.rag_system.sources.length > 0 && (
                    <View style={styles.sourcesContainer}>
                      <Text style={styles.sourceHeader}>Verified Sources:</Text>
                      {result.rag_system.sources.map((src, idx) => (
                        <Text key={idx} style={styles.sourceText}>• {src.title}</Text>
                      ))}
                    </View>
                  )}
                </View>
              </View>
            </View>
          )}
        </ScrollView>

        {/* Input Area */}
        <View style={styles.inputContainer}>
          <TextInput
            style={styles.textInput}
            value={inputText}
            onChangeText={setInputText}
            placeholder="e.g., What fertilizer for a 2yo CRIC65?"
            placeholderTextColor="#888"
            multiline
            maxLength={200}
          />
          <TouchableOpacity
            style={[styles.sendButton, (!inputText.trim() || isLoading) && styles.sendButtonDisabled]}
            onPress={compareAnswers}
            disabled={!inputText.trim() || isLoading}
          >
            {isLoading ? (
              <ActivityIndicator size="small" color="#FFFFFF" />
            ) : (
              <Ionicons name="search" size={20} color="#FFFFFF" />
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
    backgroundColor: '#F5F9F6',
  },
  header: {
    backgroundColor: '#1E3A8A', // Deep Blue for comparison theme
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
    color: '#E0E7FF',
    fontSize: 14,
    textAlign: 'center',
    fontWeight: '500',
    opacity: 0.9,
  },
  keyboardAvoid: {
    flex: 1,
  },
  scrollContainer: {
    padding: 16,
    paddingBottom: 32,
    flexGrow: 1,
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 100,
  },
  emptyText: {
    marginTop: 16,
    fontSize: 16,
    color: '#888',
    textAlign: 'center',
    paddingHorizontal: 32,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 100,
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#555',
    fontWeight: '600',
  },
  errorContainer: {
    backgroundColor: '#FFEBEE',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#FFCDD2',
  },
  errorText: {
    color: '#C62828',
    fontSize: 14,
  },
  resultsContainer: {
    gap: 16,
  },
  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 6,
    elevation: 3,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    paddingHorizontal: 16,
  },
  plainHeader: {
    backgroundColor: '#FFF3E0',
    borderBottomWidth: 1,
    borderBottomColor: '#FFE0B2',
  },
  ragHeader: {
    backgroundColor: '#E8F5E9',
    borderBottomWidth: 1,
    borderBottomColor: '#C8E6C9',
  },
  plainHeaderText: {
    color: '#E65100',
    fontWeight: '700',
    fontSize: 16,
    marginLeft: 8,
  },
  ragHeaderText: {
    color: '#1B5E20',
    fontWeight: '700',
    fontSize: 16,
    marginLeft: 8,
  },
  cardBody: {
    padding: 16,
  },
  answerText: {
    fontSize: 15,
    lineHeight: 24,
    color: '#333333',
  },
  warningNote: {
    marginTop: 12,
    fontSize: 12,
    color: '#E65100',
    fontStyle: 'italic',
    fontWeight: '600',
  },
  contextBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#E0E7FF',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#C7D2FE',
  },
  contextBannerText: {
    marginLeft: 8,
    color: '#1E3A8A',
    fontSize: 14,
    fontWeight: '600',
  },
  contextUsedContainer: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#EEEEEE',
  },
  contextUsedText: {
    fontSize: 12,
    color: '#1B5E20',
    fontStyle: 'italic',
  },
  sourcesContainer: {
    marginTop: 16,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#EEEEEE',
  },
  sourceHeader: {
    fontSize: 13,
    fontWeight: '700',
    color: '#1B5E20',
    marginBottom: 6,
  },
  sourceText: {
    fontSize: 13,
    color: '#666666',
    marginTop: 2,
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 16,
    backgroundColor: '#FFFFFF',
    borderTopWidth: 1,
    borderTopColor: '#EEEEEE',
    alignItems: 'center',
  },
  textInput: {
    flex: 1,
    backgroundColor: '#F5F5F5',
    borderRadius: 24,
    paddingHorizontal: 20,
    paddingTop: 14,
    paddingBottom: 14,
    fontSize: 16,
    maxHeight: 100,
    color: '#333',
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  sendButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#1E3A8A',
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 12,
  },
  sendButtonDisabled: {
    backgroundColor: '#93C5FD',
  },
});
