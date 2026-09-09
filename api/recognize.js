/**
 * Vercel Serverless Function - Recipe Recognition Orchestrator
 * Endpoint: https://your-vercel-domain.vercel.app/api/recognize
 */

import { createClient } from "@supabase/supabase-js";
import vision from "@google-cloud/vision";
import { v4 as uuidv4 } from "uuid";

// Initialize clients
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

const visionClient = new vision.ImageAnnotatorClient({
  keyFilename: process.env.GOOGLE_CLOUD_KEY_PATH,
});

// Ingredient mapping (same as Python)
const INGREDIENT_MAPPING = {
  tomato: { normalized_id: "ingredient_tomato_001", display_name: "Tomate" },
  "olive oil": {
    normalized_id: "ingredient_olive_oil_001",
    display_name: "Olivenöl",
  },
  garlic: { normalized_id: "ingredient_garlic_001", display_name: "Knoblauch" },
  pasta: { normalized_id: "ingredient_pasta_001", display_name: "Pasta" },
  onion: { normalized_id: "ingredient_onion_001", display_name: "Zwiebel" },
  chicken: { normalized_id: "ingredient_chicken_001", display_name: "Hähnchen" },
  mushroom: {
    normalized_id: "ingredient_mushroom_001",
    display_name: "Champignon",
  },
  "bell pepper": {
    normalized_id: "ingredient_bell_pepper_001",
    display_name: "Paprika",
  },
  cheese: { normalized_id: "ingredient_cheese_001", display_name: "Käse" },
  salt: { normalized_id: "ingredient_salt_001", display_name: "Salz" },
  pepper: { normalized_id: "ingredient_pepper_001", display_name: "Pfeffer" },
  butter: { normalized_id: "ingredient_butter_001", display_name: "Butter" },
};

const CONFIDENCE_THRESHOLD = parseFloat(
  process.env.CONFIDENCE_THRESHOLD || "0.65"
);

// ============================================
// ORCHESTRATOR CLASS
// ============================================

class RecipeOrchestrator {
  constructor() {
    this.taskId = uuidv4();
    this.traceId = this.taskId;
    this.orchestrationState = {
      image_handler_status: "pending",
      recognizer_status: "pending",
      recipe_status: "pending",
      overall_status: "pending",
    };
    this.errorLog = [];
  }

  log(eventType, details) {
    const logEntry = {
      timestamp: new Date().toISOString(),
      trace_id: this.traceId,
      task_id: this.taskId,
      event_type: eventType,
      details,
    };
    console.log(JSON.stringify(logEntry));
    return logEntry;
  }

  // ============================================
  // PHASE 1: Image Handler
  // ============================================
  validateImage(imageBase64) {
    this.log("IMAGE_VALIDATION_START", {});

    try {
      // Decode base64
      const imageBuffer = Buffer.from(imageBase64, "base64");

      // Size check (max 5MB)
      if (imageBuffer.length > 5 * 1024 * 1024) {
        throw new Error("Image size exceeds 5MB limit");
      }

      // Format check
      const isValidFormat = this._isValidImageFormat(imageBuffer);
      if (!isValidFormat) {
        throw new Error("Invalid image format (jpeg/png/heic required)");
      }

      this.orchestrationState.image_handler_status = "completed";

      this.log("IMAGE_VALIDATION_SUCCESS", {
        size_bytes: imageBuffer.length,
        format: this._detectFormat(imageBuffer),
      });

      return {
        processed_image: imageBase64,
        image_metadata: {
          size_bytes: imageBuffer.length,
          quality_score: 0.95,
        },
        validation_passed: true,
      };
    } catch (error) {
      this.orchestrationState.image_handler_status = "failed";
      const errorMsg = `Image validation failed: ${error.message}`;
      this.errorLog.push(errorMsg);
      this.log("IMAGE_VALIDATION_FAILED", { error: errorMsg });
      throw error;
    }
  }

  _isValidImageFormat(buffer) {
    // JPEG
    if (buffer.subarray(0, 3).toString() === "255,216,255") return true;
    // PNG
    if (buffer.subarray(0, 8).toString() === "137,80,78,71,13,10,26,10")
      return true;
    // HEIC
    if (buffer.subarray(0, 4).toString() === "0,0,0,24") return true;
    return false;
  }

  _detectFormat(buffer) {
    if (buffer.subarray(0, 3).toString() === "255,216,255") return "jpeg";
    if (buffer.subarray(0, 8).toString() === "137,80,78,71,13,10,26,10")
      return "png";
    if (buffer.subarray(0, 4).toString() === "0,0,0,24") return "heic";
    return "unknown";
  }

  // ============================================
  // PHASE 2: Ingredient Recognizer (Google Vision)
  // ============================================
  async recognizeIngredients(imageBase64) {
    this.log("INGREDIENT_RECOGNITION_START", {});

    try {
      const imageBuffer = Buffer.from(imageBase64, "base64");

      // Call Google Vision API
      const request = {
        image: {
          content: imageBuffer,
        },
      };

      const [result] = await visionClient.labelDetection(request);
      const labels = result.labelAnnotations;

      this.log("VISION_API_SUCCESS", {
        label_count: labels.length,
      });

      // Process and normalize labels
      const ingredients = this._processLabels(labels);

      // Filter by confidence
      const filteredIngredients = ingredients.filter(
        (ing) => ing.confidence >= CONFIDENCE_THRESHOLD
      );

      const overallConfidence =
        filteredIngredients.length > 0
          ? filteredIngredients.reduce((sum, ing) => sum + ing.confidence, 0) /
            filteredIngredients.length
          : 0;

      this.orchestrationState.recognizer_status = "completed";

      this.log("INGREDIENT_RECOGNITION_SUCCESS", {
        ingredient_count: filteredIngredients.length,
        confidence: overallConfidence,
      });

      return {
        recognized_ingredients: filteredIngredients,
        overall_confidence: overallConfidence,
        recognition_quality:
          overallConfidence > 0.8
            ? "high"
            : overallConfidence > 0.6
              ? "medium"
              : "low",
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      this.orchestrationState.recognizer_status = "failed";
      const errorMsg = `Ingredient recognition failed: ${error.message}`;
      this.errorLog.push(errorMsg);
      this.log("INGREDIENT_RECOGNITION_FAILED", { error: errorMsg });
      throw error;
    }
  }

  _processLabels(labels) {
    const ingredients = [];
    const seenIds = new Set();

    for (const label of labels) {
      const labelName = label.description.toLowerCase();
      const confidence = label.score;

      // Skip low confidence
      if (confidence < CONFIDENCE_THRESHOLD) continue;

      const normalized = this._normalizeIngredient(labelName, confidence);

      if (
        normalized &&
        !seenIds.has(normalized.normalized_id)
      ) {
        ingredients.push(normalized);
        seenIds.add(normalized.normalized_id);
      }
    }

    return ingredients;
  }

  _normalizeIngredient(labelName, confidence) {
    // Direct match
    if (labelName in INGREDIENT_MAPPING) {
      const mapping = INGREDIENT_MAPPING[labelName];
      return {
        name: labelName,
        display_name: mapping.display_name,
        normalized_id: mapping.normalized_id,
        confidence: confidence,
        source: "direct_match",
      };
    }

    // Fuzzy match
    for (const [key, mapping] of Object.entries(INGREDIENT_MAPPING)) {
      if (key.includes(labelName) || labelName.includes(key)) {
        return {
          name: key,
          display_name: mapping.display_name,
          normalized_id: mapping.normalized_id,
          confidence: confidence * 0.95,
          source: "fuzzy_match",
        };
      }
    }

    return null;
  }

  // ============================================
  // PHASE 3: Recipe Retrieval
  // ============================================
  async retrieveRecipes(ingredients) {
    this.log("RECIPE_RETRIEVAL_START", {
      ingredient_count: ingredients.length,
    });

    try {
      const ingredientIds = ingredients.map((ing) => ing.normalized_id);

      // Query Supabase
      const { data: recipes, error } = await supabase.rpc(
        "find_recipes_by_ingredients",
        {
          ingredient_ids: ingredientIds,
          max_results: 5,
        }
      );

      if (error) throw error;

      this.orchestrationState.recipe_status = "completed";

      this.log("RECIPE_RETRIEVAL_SUCCESS", {
        recipe_count: recipes.length,
      });

      return {
        recipes: recipes || [],
        total_results: recipes.length,
        search_timestamp: new Date().toISOString(),
      };
    } catch (error) {
      this.orchestrationState.recipe_status = "failed";
      const errorMsg = `Recipe retrieval failed: ${error.message}`;
      this.errorLog.push(errorMsg);
      this.log("RECIPE_RETRIEVAL_FAILED", { error: errorMsg });

      // Fallback: return empty instead of failing entire request
      return {
        recipes: [],
        total_results: 0,
        search_timestamp: new Date().toISOString(),
        fallback: true,
      };
    }
  }

  // ============================================
  // PHASE 4: Result Synthesizer
  // ============================================
  synthesizeResponse(ingredients, recipes) {
    this.log("SYNTHESIS_START", {});

    try {
      const ingredientConfidence =
        ingredients.length > 0
          ? ingredients.reduce((sum, ing) => sum + ing.confidence, 0) /
            ingredients.length
          : 0;

      const overallConfidence = ingredientConfidence;

      // Determine status
      let status = "success";
      if (overallConfidence < 0.65 || recipes.length === 0) {
        status = "degraded";
      }
      if (overallConfidence < 0.5) {
        status = "failed";
      }

      this.orchestrationState.overall_status = status;

      const response = {
        task_id: this.taskId,
        status: status,
        ingredients: ingredients,
        recipes: recipes,
        quality_metrics: {
          ingredient_confidence: ingredientConfidence,
          recipe_count: recipes.length,
          overall_confidence: overallConfidence,
        },
        orchestration_state: this.orchestrationState,
        error_log: this.errorLog,
        timestamp: new Date().toISOString(),
      };

      this.log("SYNTHESIS_SUCCESS", {
        status: status,
        confidence: overallConfidence,
      });

      return response;
    } catch (error) {
      const errorMsg = `Synthesis failed: ${error.message}`;
      this.errorLog.push(errorMsg);
      this.log("SYNTHESIS_FAILED", { error: errorMsg });
      throw error;
    }
  }

  // ============================================
  // Main Orchestration
  // ============================================
  async process(event) {
    this.log("REQUEST_START", {
      operation: event.operation,
    });

    try {
      const imageData = event.image_data;
      if (!imageData) {
        throw new Error("image_data is required");
      }

      // Phase 1: Validate image
      this.validateImage(imageData);

      // Phase 2: Recognize ingredients
      const ingredientsResult = await this.recognizeIngredients(imageData);
      const ingredients = ingredientsResult.recognized_ingredients;

      // Phase 3: Retrieve recipes
      const recipesResult = await this.retrieveRecipes(ingredients);
      const recipes = recipesResult.recipes;

      // Phase 4: Synthesize response
      const finalResponse = this.synthesizeResponse(ingredients, recipes);

      return {
        statusCode: 200,
        body: JSON.stringify(finalResponse),
        headers: {
          "Content-Type": "application/json",
        },
      };
    } catch (error) {
      this.log("ORCHESTRATION_FAILED", {
        error: error.message,
      });

      return {
        statusCode: 400,
        body: JSON.stringify({
          task_id: this.taskId,
          status: "failed",
          error: error.message,
          error_log: this.errorLog,
        }),
        headers: {
          "Content-Type": "application/json",
        },
      };
    }
  }
}

// ============================================
// VERCEL HANDLER
// ============================================

export default async function handler(req, res) {
  // CORS headers
  res.setHeader("Access-Control-Allow-Credentials", "true");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader(
    "Access-Control-Allow-Methods",
    "GET,OPTIONS,PATCH,DELETE,POST,PUT"
  );
  res.setHeader(
    "Access-Control-Allow-Headers",
    "X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version"
  );

  if (req.method === "OPTIONS") {
    res.status(200).end();
    return;
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const orchestrator = new RecipeOrchestrator();
  const result = await orchestrator.process(req.body);

  return res
    .status(result.statusCode)
    .setHeader("Content-Type", "application/json")
    .end(result.body);
}
