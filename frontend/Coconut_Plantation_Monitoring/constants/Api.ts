/**
 * API Configuration Constants
 * Update the BASE_URL whenever your machine's IP address changes.
 */

// Your current machine IP on the phone's hotspot
export const BASE_IP = '10.165.254.87';
export const BASE_PORT = '8000';
export const BASE_URL = `http://${BASE_IP}:${BASE_PORT}`;

export const API_ENDPOINTS = {
  ASK: `${BASE_URL}/ask`,
  COMPARE: `${BASE_URL}/compare`,
  HEALTH: `${BASE_URL}/health`,
};
